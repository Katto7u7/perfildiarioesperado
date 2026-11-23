"""Monitoreo de la operación (docs/05 §0 y §5): predicciones guardadas vs lo realizado + salud de las corridas.

  python -m perfil_retiro.monitoreo [--dias 20] [--db RUTA]       (por defecto: la base destino de config.yaml)
Salida: out/monitoreo/monitoreo_AAAAMMDD.md. Código de salida: 0 = todo en rango | 3 = alguna alerta.

Reglas:
- Solo lectura sobre la base destino.
- Lo "realizado" usa EXACTAMENTE las reglas de validez del pipeline (panel limpio: se excluyen días COPIA/PARCIAL,
  duplicados y sucursales sin evidencia; una sucursal sin fila en un día cargado cuenta como retiro 0).
- Condiciones de salida de la sombra (criterios que fallaron en el holdout, EV-13), con >= 20 días objetivo:
  1 |sesgo de retiro_esperado| <= 3%   2 cobertura por sucursal >= esperado con calibración perfecta - 5 pp
  3 P(crítico al cierre) con error <= 5 pp en todos los deciles   4 WAPE <= 1.15 x 0.352
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .calidad import detectar_copias, detectar_parciales, mov_diario
from .config import RAIZ, cargar_config
from .evaluacion import calibracion_prob
from .panel import calendario_operativo, construir_panel
from .persistencia import ruta_destino
from .sources import extraer

REF = {"wape_dev": 0.319, "wape_holdout": 0.352}
MIN_DIAS_CRITERIO = 20


def realizado_valido(cfg, desde: int) -> pd.DataFrame:
    """Panel limpio (mismas reglas que el pipeline) desde `desde` hasta el último día cargado en MOVIMIENTOS."""
    mv = extraer(cfg, "movimientos_diario", desde, 29991231, usar_cache=False)
    if mv.empty:
        return pd.DataFrame()
    hasta = int(mv.id_fecha.max())
    mon = extraer(cfg, "monitoreo_cortes", desde, hasta, usar_cache=False)
    cal = calendario_operativo(extraer(cfg, "fecha", usar_cache=False), extraer(cfg, "dispersion", usar_cache=False))
    dia = mov_diario(mv)
    copias = set(detectar_copias(dia).id_fecha) | set(cfg["calidad"]["fechas_copia"])
    par = detectar_parciales(dia, mon[mon.hora_corte == 1700])
    parciales = set(par[par.parcial].id_fecha) | set(cfg["calidad"]["fechas_parciales"]) | {20251209}
    p = construir_panel(mv, mon, cal, cfg, hasta, copias=copias, parciales=parciales, desde=desde)
    return p[p.valido]


def eventos_realizados(e: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crítico al cierre y quiebre intradía REALIZADOS, sin la dotación del día (mismas definiciones que la predicción)."""
    s0 = e.sdo_inicial
    completo = e[[f"retiros_{c}" for c in (900, 1100, 1300, 1700)]].notna().all(axis=1) & s0.notna()
    q = np.zeros(len(e), bool)
    for c in (900, 1100, 1300):
        sc = s0 + e[f"depositos_{c}"] - e[f"dotaciones_{c}"] - e[f"retiros_{c}"]
        q |= (sc < (1 - e[f"frac_{c}"]) * e.umbral_critico).fillna(False).values
    cierre = s0 + e.depositos_1700 - e.dotaciones_1700 - e.retiros_1700
    q |= (cierre < 0).fillna(False).values
    crit = (cierre < e.umbral_critico).fillna(False).values
    return completo.values, crit.astype(float), q.astype(float)


def salud_corridas(con, cal_ops: np.ndarray) -> tuple[dict, list[str]]:
    """Atraso de corridas, estabilidad de tipo_perfil, cobertura del perfil y confianza (corrida vs la anterior)."""
    ind, alertas = {}, []
    runs = pd.read_sql("SELECT id_politica_run, fecha_corte, regimen, n_perfiles, modo FROM PERFIL_RUN WHERE vigente = 1 "
                       "ORDER BY fecha_corte", con)
    if runs.empty:
        return ind, ["No hay corridas vigentes en la base destino"]
    ult = runs.iloc[-1]
    hoy = int(dt.date.today().strftime("%Y%m%d"))
    pend = [d for d in cal_ops if ult.fecha_corte < d < hoy]
    ind["ultima_fecha_corte"] = int(ult.fecha_corte)
    ind["dias_operativos_sin_corrida"] = len(pend)
    if len(pend) > 1:
        alertas.append(f"Corridas atrasadas: la última fecha de corte es {ult.fecha_corte} y hay {len(pend)} días operativos "
                       "posteriores sin corrida (¿MOVIMIENTOS sin cargar o tarea programada detenida?)")
    prev = runs[(runs.regimen == ult.regimen) & (runs.fecha_corte < ult.fecha_corte)]
    if len(prev):
        ant = prev.iloc[-1]
        a = pd.read_sql("SELECT id_sucursal, tipo_perfil FROM PERFIL_RETIRO_SUCURSAL WHERE id_politica_run = ?", con,
                        params=[ult.id_politica_run]).set_index("id_sucursal").tipo_perfil
        b = pd.read_sql("SELECT id_sucursal, tipo_perfil FROM PERFIL_RETIRO_SUCURSAL WHERE id_politica_run = ?", con,
                        params=[ant.id_politica_run]).set_index("id_sucursal").tipo_perfil
        comunes = a.index.intersection(b.index)
        cambio = float((a[comunes] != b[comunes]).mean()) if len(comunes) else np.nan
        dn = (ult.n_perfiles - ant.n_perfiles) / max(ant.n_perfiles, 1)
        ind["corrida_comparada"] = f"{ult.fecha_corte} vs {ant.fecha_corte} (régimen {ult.regimen})"
        ind["pct_cambio_tipo_perfil"] = cambio
        ind["cambio_n_perfiles"] = float(dn)
        if cambio > 0.20:
            alertas.append(f"Estabilidad: {cambio:.1%} de sucursales cambió de tipo_perfil (> 20%; backtest 9-14%)")
        if abs(dn) > 0.03:
            alertas.append(f"Cobertura del perfil: n_perfiles cambió {dn:+.1%} (> 3%)")
    conf = pd.read_sql("""SELECT r.fecha_corte, p.nivel_confianza, COUNT(*) n FROM PREDICCION_RETIRO_DIARIA p
                          JOIN PERFIL_RUN r USING (id_politica_run) WHERE r.vigente = 1 GROUP BY 1, 2""", con)
    if conf.fecha_corte.nunique() >= 1:
        t = conf.pivot_table(index="fecha_corte", columns="nivel_confianza", values="n", fill_value=0)
        baja = (t.reindex(columns=["BAJA", "INSUFICIENTE"], fill_value=0).sum(axis=1) / t.sum(axis=1)).sort_index()
        ind["pct_confianza_baja_o_insuficiente"] = float(baja.iloc[-1])
        if len(baja) >= 2 and baja.iloc[-1] - baja.iloc[-2] > 0.05:
            alertas.append(f"Confianza: BAJA+INSUFICIENTE subió de {baja.iloc[-2]:.1%} a {baja.iloc[-1]:.1%} (> 5 pp)")
    return ind, alertas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=20)
    ap.add_argument("--db", type=Path, default=None, help="por defecto: destino.sqlite de config.yaml")
    a = ap.parse_args(argv)
    cfg = cargar_config()
    db = Path(a.db) if a.db else ruta_destino(cfg)
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    cal = calendario_operativo(extraer(cfg, "fecha", usar_cache=False), extraer(cfg, "dispersion", usar_cache=False))
    ind_salud, alertas = salud_corridas(con, cal[cal.operativo].id_fecha.values)
    pr = pd.read_sql("""SELECT p.* FROM PREDICCION_RETIRO_DIARIA p JOIN PERFIL_RUN r USING (id_politica_run)
                        WHERE r.vigente = 1""", con)
    con.close()
    ind, criterios = {"base": str(db)}, []
    fechas = np.array([], dtype=int)
    if not pr.empty:
        real = realizado_valido(cfg, int(pr.fecha_objetivo.min()))
        cols = ["id_fecha", "id_sucursal", "retiro", "sdo_inicial"] + [f"{v}_{c}" for v in ("retiros", "depositos", "dotaciones")
                                                                       for c in (900, 1100, 1300, 1700)]
        e = pr.merge(real[cols].rename(columns={"id_fecha": "fecha_objetivo", "retiro": "real"}),
                     on=["fecha_objetivo", "id_sucursal"], how="inner") if len(real) else pd.DataFrame()
        if len(e):
            fechas = np.sort(e.fecha_objetivo.unique())[-a.dias:]
            e = e[e.fecha_objetivo.isin(fechas)]
    if len(fechas) == 0:
        ind["dias_evaluados"] = 0
        txt_pred = ["Aún no hay días objetivo realizados y válidos para evaluar las predicciones."]
    else:
        ins = (e.real >= e.retiro_p10) & (e.real <= e.retiro_p90)
        cs = ins.groupby(e.id_sucursal).agg(["mean", "size"])
        n = cs["size"]
        esperado = float(np.mean([stats.binom.cdf(np.floor(0.9 * k), k, 0.8) - stats.binom.cdf(np.ceil(0.7 * k) - 1, k, 0.8)
                                  for k in n]))
        obs_suc = float(((cs["mean"] >= 0.7) & (cs["mean"] <= 0.9)).mean())
        wape = float((e.retiro_p50 - e.real).abs().sum() / e.real.sum())
        sesgo = float((e.retiro_esperado - e.real).sum() / e.real.sum())
        completo, crit, quie = eventos_realizados(e)
        okc = completo & e.prob_saldo_critico.notna().values
        okq = completo & e.prob_quiebre_intradia.notna().values
        tc, mc = calibracion_prob(e.prob_saldo_critico.values[okc], crit[okc], bins=10) if okc.sum() >= 100 else (None, {})
        tq, mq = calibracion_prob(e.prob_quiebre_intradia.values[okq], quie[okq], bins=10) if okq.sum() >= 100 else (None, {})
        err_crit = float(tc.error_abs_pp.max()) if tc is not None else np.nan
        ind.update({"dias_evaluados": int(len(fechas)), "primer_dia": int(fechas.min()), "ultimo_dia": int(fechas.max()),
                    "pronosticos": int(len(e)), "WAPE_p50": wape, "sesgo_retiro_esperado": sesgo,
                    "cobertura_P10_P90": float(ins.mean()), "pct_suc_cobertura_70_90": obs_suc,
                    "pct_esperado_con_calibracion_perfecta": esperado,
                    "critico_cierre_error_max_decil_pp": err_crit, "critico_cierre_auc": mc.get("auc", np.nan),
                    "critico_cierre_tasa_real": mc.get("tasa_base", np.nan),
                    "quiebre_error_max_decil_pp": float(tq.error_abs_pp.max()) if tq is not None else np.nan,
                    "quiebre_auc": mq.get("auc", np.nan), "quiebre_tasa_real": mq.get("tasa_base", np.nan)})
        suf = len(fechas) >= MIN_DIAS_CRITERIO
        for nom, val, ok, umbral in (
                ("1 |sesgo de retiro_esperado|", abs(sesgo), abs(sesgo) <= 0.03, "<= 3%"),
                ("2 cobertura por sucursal", obs_suc, obs_suc >= esperado - 0.05, f">= {esperado - 0.05:.1%}"),
                ("3 error máx. por decil P(crítico cierre)", err_crit / 100, (err_crit <= 5) if not np.isnan(err_crit) else False, "<= 5 pp"),
                ("4 WAPE", wape, wape <= 1.15 * REF["wape_holdout"], f"<= {1.15 * REF['wape_holdout']:.3f}")):
            criterios.append((nom, val, umbral, ("CUMPLE" if ok else "NO CUMPLE") if suf else f"INSUFICIENTE (<{MIN_DIAS_CRITERIO} días)"))
        if wape > 1.15 * REF["wape_holdout"]:
            alertas.append(f"WAPE {wape:.3f} > 1.15 × referencia del holdout ({REF['wape_holdout']})")
        if len(fechas) >= 10 and abs(sesgo) > 0.03:
            alertas.append(f"|sesgo de retiro_esperado| {sesgo:+.1%} > 3%")
        if len(fechas) >= 5 and not 0.72 <= ind["cobertura_P10_P90"] <= 0.88:     # 1-4 días: ruido, no señal
            alertas.append(f"cobertura P10-P90 {ind['cobertura_P10_P90']:.1%} fuera de [72%, 88%]")
        if n.median() >= 10 and obs_suc < esperado - 0.05:
            alertas.append(f"cobertura por sucursal {obs_suc:.1%} < esperado con calibración perfecta ({esperado:.1%}) − 5 pp")
        txt_pred = []
    ind.update(ind_salud)
    d = RAIZ / "out" / "monitoreo"
    d.mkdir(parents=True, exist_ok=True)
    hoy = dt.date.today().strftime("%Y%m%d")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    txt = [f"# Monitoreo — {hoy}", "", *txt_pred, "| Indicador | Valor |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in ind.items()]
    if criterios:
        txt += ["", "## Condiciones para salir de la sombra", "| Criterio | Valor | Umbral | Resultado |", "|---|---|---|---|"]
        txt += [f"| {a_} | {v:.4f} | {u} | {r} |" for a_, v, u, r in criterios]
    txt += ["", "## Alertas", *(f"- {x}" for x in alertas)] if alertas else ["", "## Alertas", "- Ninguna: todos los indicadores en rango."]
    (d / f"monitoreo_{hoy}.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    return 3 if alertas else 0


if __name__ == "__main__":
    sys.exit(main())
