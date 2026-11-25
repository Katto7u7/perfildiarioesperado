"""Corrida diaria (nocturna): perfil + predicción del siguiente día operativo.

Pasos (cada uno registrado en el log y en PERFIL_RUN):
 1 extracción (solo lectura, id_fecha <= fecha_corte)   2 control de calidad del día de corte (aborta si dañado)
 3 panel limpio   4 pronóstico (M1, NB, M2 -> ENS_ALL)   5 cuantiles conformal (errores fuera de muestra guardados)
 6 riesgo de saldo y confianza   7 perfil   8 contrato   9 persistencia transaccional e idempotente
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from .backtest import CORTES, cdf_desde_cuantiles, eventos_pe, forma_esperada, letras_por_fecha, marcar_evento
from .calidad import detectar_copias, detectar_parciales, mov_diario
from .contrato import validar_corte, validar_perfil
from .modelo import M1, QUANTS, baseline_nb
from .panel import calendario_operativo, construir_panel
from .perfil import calcular_perfil
from .persistencia import (ahora_utc, conectar_destino, conectar_dev, escribir, id_run, respaldar, ruta_destino,
                           sucursales_catalogo)
from .sources import extraer, huella_insumos

log = logging.getLogger("perfil_retiro")
QC = [f"q{int(a*100):02d}" for a in QUANTS]


class CorridaAbortada(RuntimeError):
    pass


def fecha_entrenamiento_m2(ops: pd.Series, fecha_corte: int) -> int:
    """Corte de entrenamiento de M2 = primer día operativo del mes de la fecha de corte.

    Es exactamente el esquema validado en el backtest (m2_gbm.predecir_backtest: reentrenamiento en el primer origen
    de cada mes). Es determinista y siempre <= fecha_corte: reprocesar cualquier día nunca usa datos posteriores a él,
    sin importar en qué orden se ejecuten las corridas.
    """
    o = pd.Series(ops).astype(int)
    return int(o[(o // 100) == fecha_corte // 100].min())


def cobertura_monitoreo_1700(dia_d: pd.DataFrame, mon17_d: pd.DataFrame) -> float:
    """Fracción de sucursales con retiro en D (MOVIMIENTOS) que tienen corte de las 17:00 en MONITOREO_CORTE."""
    con_retiro = set(dia_d.loc[dia_d.retiro > 0, "id_sucursal"])
    if not con_retiro:
        return 0.0
    return len(con_retiro & set(mon17_d.id_sucursal)) / len(con_retiro)


def _conformal_un_origen(punto: pd.DataFrame, errores: pd.DataFrame, n_suc=120, n0=10.0):
    """Cuantiles y media para un origen a partir de errores log fuera de muestra (id_sucursal, estrato, err)."""
    # Igual que calibracion.conformal_causal (método validado en desarrollo y holdout): cuantiles globales con los
    # últimos 60,000 errores por estrato; por sucursal, los últimos n_suc.
    errores = errores.sort_values(["id_fecha", "id_sucursal"], kind="mergesort")
    eg = errores.groupby("estrato").tail(60000)
    qg = eg.groupby("estrato").err.quantile(QUANTS).unstack()
    smg = np.exp(eg.err.clip(upper=eg.err.quantile(0.999))).groupby(eg.estrato).mean()
    es = errores.sort_values("id_fecha").groupby(["id_sucursal", "estrato"]).tail(n_suc)
    qs = es.groupby(["id_sucursal", "estrato"]).err.quantile(QUANTS).unstack()
    ns = es.groupby(["id_sucursal", "estrato"]).size()
    sms = np.exp(es.err.clip(upper=3)).groupby([es.id_sucursal, es.estrato]).mean()
    idx = pd.MultiIndex.from_arrays([punto.id_sucursal, punto.estrato])
    nn = ns.reindex(idx).fillna(0).values
    w = (nn / (nn + n0))[:, None]
    Q = np.where(np.isnan(qs.reindex(idx).values), qg.reindex(punto.estrato).values,
                 w * qs.reindex(idx).values + (1 - w) * qg.reindex(punto.estrato).values)
    Q = np.maximum.accumulate(Q, axis=1)
    lp = np.log1p(punto.punto.values)
    out = pd.DataFrame({c: np.expm1(lp + Q[:, i]).clip(min=0) for i, c in enumerate(QC)}, index=punto.index)
    sm = np.where(np.isnan(sms.reindex(idx).values), smg.reindex(punto.estrato).values,
                  w[:, 0] * sms.reindex(idx).values + (1 - w[:, 0]) * smg.reindex(punto.estrato).values)
    if {"real_ok", "pt"} <= set(errores.columns):        # estimador de razón (EV-12): Σreal/Σpronóstico, 40 días
        fechas = np.sort(errores.id_fecha.unique())[-40:]
        er = errores[errores.id_fecha.isin(fechas)]
        Rg = er.groupby("estrato").real_ok.sum() / er.groupby("estrato").pt.sum()
        den = es.groupby(["id_sucursal", "estrato"]).pt.sum()
        Rs = (es.groupby(["id_sucursal", "estrato"]).real_ok.sum() / den.where(den > 0)).clip(0.2, 5)
        rs, rg = Rs.reindex(idx).values, Rg.reindex(punto.estrato).values
        wr = w[:, 0] * 0.5
        out["media"] = (punto.punto.values * np.where(np.isnan(rs), rg, wr * rs + (1 - wr) * rg)).clip(min=0)
    else:
        out["media"] = (np.exp(lp) * sm - 1).clip(min=0)
    out["n_errores_suc"] = nn
    return out


def _programas_emergentes(cfg) -> pd.DataFrame | None:
    """PROGRAMAS_EMERGENTES (LEEN/LCEN): fuente opcional. Solo se usa lo solicitado <= fecha de corte."""
    try:
        return extraer(cfg, "programas_emergentes", usar_cache=False)
    except Exception as e:  # noqa: BLE001
        log.warning("PROGRAMAS_EMERGENTES no disponible: %s", e)
        return None


def errores_realizados(db_path, valid: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Errores fuera de muestra de corridas anteriores en la base `db_path` (la misma a la que se escribe):
    predicción vigente guardada vs retiro real (solo días válidos del panel)."""
    import sqlite3
    from pathlib import Path
    path = db_path
    try:
        con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        pr = pd.read_sql("""SELECT p.id_sucursal, p.fecha_objetivo AS id_fecha, p.retiro_p50 AS pt, p.retiro_p10, p.retiro_p90
                            FROM PREDICCION_RETIRO_DIARIA p JOIN PERFIL_RUN r USING (id_politica_run) WHERE r.vigente = 1""", con)
        con.close()
    except Exception as e:  # base sin tablas aún (primera corrida): se usa solo la semilla
        log.warning("Sin errores propios guardados (%s): se calibra solo con la semilla", e)
        return pd.DataFrame()
    if pr.empty:
        return pr
    real = valid[["id_sucursal", "id_fecha", "retiro"]].rename(columns={"retiro": "real"})
    e = pr.merge(real, on=["id_sucursal", "id_fecha"], how="inner")
    am = cal.set_index("id_fecha").disp_am.fillna(False).astype(bool)
    e["estrato"] = np.where(e.id_fecha.map(am).fillna(False), "AM", "NO_AM")
    e["err"] = np.log1p(e.real) - np.log1p(e.pt)
    e["real_ok"], e["abs_err"] = e.real, (e.pt - e.real).abs()
    e["dentro"] = ((e.real >= e.retiro_p10) & (e.real <= e.retiro_p90)).astype(float)
    return e[["id_sucursal", "estrato", "id_fecha", "err", "real", "pt", "real_ok", "abs_err", "dentro"]]


def nivel_confianza(err_suc: pd.DataFrame, n_dias: pd.Series, cfg) -> pd.Series:
    """ALTA/MEDIA/BAJA/INSUFICIENTE con evidencia fuera de muestra de los últimos 60 días (docs/02 §4)."""
    e = err_suc.sort_values("id_fecha").groupby("id_sucursal").tail(60)
    g = e.groupby("id_sucursal")
    wape = g.apply(lambda t: (t.abs_err.sum() / max(t.real.sum(), 1)), include_groups=False)
    cov = g.dentro.mean()
    # CUSUM simple sobre errores log de los últimos 20 días: cambio de nivel si |media| > 2.5 sd/sqrt(n)
    last = e.groupby("id_sucursal").tail(20).groupby("id_sucursal").err.agg(["mean", "std", "size"])
    cambio = (last["mean"].abs() > 2.5 * last["std"] / np.sqrt(last["size"])) & (last["size"] >= 15)
    idx = n_dias.index
    w, c, ch = wape.reindex(idx), cov.reindex(idx), cambio.reindex(idx).fillna(False)
    nivel = np.select([(n_dias < 60) | ch | w.isna(),
                       (w <= 0.20) & c.between(0.72, 0.88),
                       (w <= 0.35) & c.between(0.65, 0.92)], ["INSUFICIENTE", "ALTA", "MEDIA"], "BAJA")
    return pd.Series(nivel, index=idx), w


def _calcular(cfg, fecha_corte: int, errores_semilla: pd.DataFrame | None, db_path, lista_8am, solo_perfil: bool,
              modo: str) -> dict:
    """Un cómputo completo (pasos 1-8) SIN escribir nada. Devuelve run y tablas listas para persistir."""
    destino = db_path or ruta_destino(cfg)
    t0 = time.time()
    ini = int(cfg["historia"]["inicio_util"])
    # 1 extracción (sin caché: datos frescos)
    src = {n: extraer(cfg, n, ini, fecha_corte, usar_cache=False) for n in ("movimientos_diario", "monitoreo_cortes")}
    fecha, disp, suc = (extraer(cfg, n, usar_cache=False) for n in ("fecha", "dispersion", "sucursal"))
    pe = _programas_emergentes(cfg)
    pe_corte = pe[pe.id_fecha_solicitud <= fecha_corte] if pe is not None else pd.DataFrame()
    # huella = TODO insumo que puede cambiar el resultado (si cambia, cambia el id_politica_run)
    huella = huella_insumos({**src, "dispersion": disp, "fecha": fecha, "sucursal": suc, "programas_emergentes": pe_corte})
    cal = calendario_operativo(fecha, disp)
    ops = cal[cal.operativo].id_fecha.sort_values()
    if fecha_corte not in set(ops):
        raise CorridaAbortada(f"{fecha_corte} no es día operativo")
    futuros = ops[ops > fecha_corte]
    if futuros.empty:
        raise CorridaAbortada(f"FECHA no tiene días operativos posteriores a {fecha_corte}: no hay día objetivo")
    T = int(futuros.iloc[0])
    avisos = []
    if int(disp.id_fecha.max()) < T:
        # No se aborta: DISPERSION tiene huecos naturales de hasta ~18 días sin pagos (verificado 2025-2026).
        avisos.append(f"DISPERSION cargada solo hasta {int(disp.id_fecha.max())} (< día objetivo {T}): si ese día hay "
                      "pago por letra y aún no está cargado, el pronóstico no lo considera")
        log.warning(avisos[-1])
    log.info("corte %s -> objetivo %s; huella %s", fecha_corte, T, json.dumps(huella))
    # 2 control de calidad del día de corte
    mv, mon = src["movimientos_diario"], src["monitoreo_cortes"]
    if mv.id_fecha.max() < fecha_corte:
        raise CorridaAbortada(f"MOVIMIENTOS sin cierre de {fecha_corte} (máx {mv.id_fecha.max()}): no se publica con datos viejos")
    dia = mov_diario(mv)
    # Sin el corte de 17:00 de D no se puede detectar una carga parcial de MOVIMIENTOS ni conocer el saldo inicial.
    cob = cobertura_monitoreo_1700(dia[dia.id_fecha == fecha_corte],
                                   mon[(mon.hora_corte == 1700) & (mon.id_fecha == fecha_corte)])
    min_cob = float(cfg["calidad"].get("min_cobertura_monitoreo_1700", 0.90))
    if cob < min_cob:
        raise CorridaAbortada(f"MONITOREO_CORTE@1700 de {fecha_corte} cubre {cob:.1%} de las sucursales con retiro "
                              f"(mínimo {min_cob:.0%}): no se puede validar la carga ni calcular el saldo inicial")
    cop = detectar_copias(dia[dia.id_fecha >= ops[ops < fecha_corte].iloc[-1]])
    par = detectar_parciales(dia[dia.id_fecha == fecha_corte], mon[(mon.hora_corte == 1700) & (mon.id_fecha == fecha_corte)])
    copias, parciales = set(cop.id_fecha), set(par[par.parcial].id_fecha)
    copias |= set(cfg["calidad"]["fechas_copia"])
    parciales |= set(cfg["calidad"]["fechas_parciales"]) | {20251209}
    calidad_d = "COPIA" if fecha_corte in copias else "PARCIAL" if fecha_corte in parciales else "OK"
    if calidad_d != "OK":
        raise CorridaAbortada(f"Día de corte {fecha_corte} con calidad {calidad_d}: se reporta y no se publica")
    # 3 panel
    p = construir_panel(mv, mon, cal, cfg, fecha_corte, copias=copias, parciales=parciales)
    valid = p[p.valido].copy()
    letras = letras_por_fecha(disp)
    valid["llave_letras"] = valid.id_fecha.map(letras).fillna("SIN")
    ev = eventos_pe(pe, fecha_corte) if pe is not None else pd.DataFrame(columns=["id_sucursal", "v_ini", "v_fin"])
    valid["evento_pe"] = marcar_evento(valid, ev)
    if errores_semilla is None or errores_semilla.empty:
        raise CorridaAbortada("Sin errores fuera de muestra para calibrar intervalos (cargar semilla del backtest)")
    pred = None
    if solo_perfil:
        errs = errores_semilla[errores_semilla.id_fecha <= fecha_corte]
        _, wape_oos = nivel_confianza(errs, valid.groupby("id_sucursal").size(), cfg)
    else:
        # 4 pronóstico
        n = valid.groupby("id_sucursal").size()
        sucs = n[n >= 20].index
        obj = pd.DataFrame({"id_sucursal": sucs, "id_fecha": T})
        cT = cal.set_index("id_fecha").loc[T]
        for c in cT.index:
            obj[c] = cT[c]
        obj["evento_pe"] = marcar_evento(obj, ev)
        obj["llave_letras"] = letras.get(T, "SIN")
        m = M1(vida_media=float(cfg["modelo"]["vida_media"])).ajustar(valid[valid.id_sucursal.isin(sucs)])
        pr = m.predecir(obj)
        pr["nb"] = baseline_nb(valid, obj, letras).values
        # M2 (gradient boosting): corte de entrenamiento = primer día operativo del mes (esquema validado); variables < T.
        # Se entrena en cada cómputo (sin caché en disco): un modelo dañado no puede propagarse entre cómputos.
        from .m2_gbm import FEATS, construir_features
        from sklearn.ensemble import HistGradientBoostingRegressor
        pT = pd.concat([p, pd.DataFrame({"id_sucursal": sucs, "id_fecha": T})], ignore_index=True)
        for c in cT.index:
            pT.loc[pT.id_fecha == T, c] = cT[c]
        pT.loc[pT.id_fecha == T, "valido"] = False
        fe = construir_features(pT, letras)
        f_m2 = fecha_entrenamiento_m2(ops, fecha_corte)
        tr = fe[fe.valido & (fe.id_fecha <= f_m2)]
        fechas_tr = np.sort(tr.id_fecha.unique())[-int(cfg["modelo"]["m2_ventana_dias"]):]
        tr = tr[tr.id_fecha.isin(fechas_tr)].dropna(subset=["f_md20"])
        m2 = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.08,
                                           max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                           random_state=int(cfg["semilla"]) % (2**31)).fit(tr[FEATS].astype(float), np.log1p(tr.retiro))
        log.info("M2 entrenado con datos <= %s (%d filas)", f_m2, len(tr))
        qT = fe[fe.id_fecha == T].set_index("id_sucursal").reindex(pr.id_sucursal)
        pr["m2"] = np.expm1(m2.predict(qT[FEATS].astype(float))).clip(min=0)
        pr["m2"] = pr.m2.fillna(pr.p50)
        am_ = pr.estrato == "AM"
        pr["punto"] = np.where(am_, (pr.p50 + pr.nb.fillna(pr.p50) + pr.m2) / 3, (pr.p50 + pr.m2) / 2)   # ENS_ALL (EV-12)
        pr["factores"] = m.explicar(obj).values
        # 5 cuantiles conformal
        propios = errores_realizados(destino, valid, cal)
        if len(propios):                       # los errores reales de la operación sustituyen a la semilla en esas fechas
            errores_semilla = pd.concat([errores_semilla[~errores_semilla.id_fecha.isin(propios.id_fecha.unique())], propios])
            log.info("errores fuera de muestra propios: %d (fechas %s-%s)", len(propios), propios.id_fecha.min(), propios.id_fecha.max())
        errs = errores_semilla[errores_semilla.id_fecha <= fecha_corte]
        q = _conformal_un_origen(pr[["id_sucursal", "estrato", "punto"]], errs)
        pr = pd.concat([pr.drop(columns=QC + ["media"]), q], axis=1)
        # 6 riesgo y confianza
        s0 = p[p.id_fecha == fecha_corte].set_index("id_sucursal").saldo_cierre.reindex(pr.id_sucursal).values
        hv = valid[valid.n_cortes == 4].sort_values("id_fecha").groupby("id_sucursal").tail(20)
        tipo = pd.Series(np.where(bool(cT.disp_am), "AM", np.where(bool(cT.disp_becas), "BECAS", "SIN")), index=pr.id_sucursal.values)
        fr = forma_esperada(valid, tipo)
        # Riesgo (definiciones del usuario 2026-11-18, EV-15), sin dotación adicional:
        #  crítico al cierre: S0 + dep_1700 − R < P90 ;  quiebre en corte c: S0 + dep_c − f_c·R < (1 − f_c)·P90 ; 17:00: saldo < 0
        U = float(cfg["perfil"]["factor_umbral_critico"]) * pr.q90.values
        qmat = pr[QC].values
        dep = {c: (hv[f"depositos_{c}"] - hv[f"dotaciones_{c}"]).groupby(hv.id_sucursal).median().reindex(pr.id_sucursal).fillna(0).values
               for c in CORTES}
        pcs = []
        for c in (900, 1100, 1300):
            f = np.maximum(fr[f"p_{c}"].values, 1e-3)
            pcs.append(1 - cdf_desde_cuantiles(qmat, (s0 + dep[c] - (1 - f) * U) / f))
        pcs.append(1 - cdf_desde_cuantiles(qmat, s0 + dep[1700]))
        pc = np.column_stack(pcs)
        p_cierre = 1 - cdf_desde_cuantiles(qmat, s0 + dep[1700] - U)
        eva = errs.copy()
        eva["abs_err"], eva["dentro"] = eva.abs_err, eva.dentro
        conf, wape_oos = nivel_confianza(eva, pr.set_index("id_sucursal").n_dias, cfg)
    # 7 perfil
    perf, corte, excl, aux = calcular_perfil(p, cal, suc, cfg, fecha_corte, wape_oos=wape_oos, lista_8am=lista_8am)
    huella["config"] = cfg.hash
    rid = id_run(fecha_corte, cfg["version_modelo"], cfg.hash, huella)
    ts = ahora_utc()
    perf["id_politica_run"], perf["created_at_utc"] = rid, ts
    corte["id_politica_run"], corte["created_at_utc"] = rid, ts
    # 8 contrato
    perf_ok, rech = validar_perfil(perf)
    corte_ok, rech_c = validar_corte(corte, set(perf_ok.id_sucursal))
    n_rech_perfil = len(rech)
    rc = rech_c[rech_c.id_sucursal.isin(perf_ok.id_sucursal)]   # los de perfiles ya rechazados no se repiten
    if len(rc):       # intervalos rechazados: el perfil se publica, pero queda registrado qué intervalo faltó y por qué
        rc = rc.assign(txt=rc.hora_inicio.astype(int).astype(str).str.zfill(4) + "-" +
                       rc.hora_fin.astype(int).astype(str).str.zfill(4) + ": " + rc.motivo.astype(str))
        rc = rc.groupby("id_sucursal").txt.apply(lambda t: "; ".join(t)).reset_index()
        rech = pd.concat([rech, pd.DataFrame({"id_sucursal": rc.id_sucursal, "motivo": "CONTRATO_INTERVALO",
                                              "detalle": rc.txt})], ignore_index=True)
    excl = pd.concat([excl, rech], ignore_index=True)
    excl["id_politica_run"] = rid
    if not solo_perfil:
        # predicciones (tabla complementaria)
        p95_perfil = perf.set_index("id_sucursal").retiro_p95_diario.reindex(pr.id_sucursal).values
        pred = pd.DataFrame({
            "id_politica_run": rid, "id_sucursal": pr.id_sucursal.values, "fecha_corte": fecha_corte, "fecha_objetivo": T,
            "horizonte_dias": int((pd.Timestamp(str(T)) - pd.Timestamp(str(fecha_corte))).days),
            "retiro_esperado": pr.media.values, "retiro_p10": pr.q10.values, "retiro_p50": pr.punto.values,
            "retiro_p90": pr.q90.values, "retiro_p95": pr.q95.values,
            "prob_retiro_extraordinario": np.where(np.isnan(p95_perfil), np.nan,
                                                   1 - cdf_desde_cuantiles(qmat, np.nan_to_num(p95_perfil, nan=0.0))),
            "saldo_inicial": s0, "umbral_critico": U, "prob_saldo_critico": np.where(np.isnan(s0), np.nan, p_cierre),
            "prob_quiebre_intradia": np.where(np.isnan(s0), np.nan, pc.max(axis=1)),
            "corte_presion": np.where(~np.isnan(s0) & (pc.max(axis=1) >= 0.5), np.array(CORTES)[np.argmax(pc >= 0.5, axis=1)], 0),
            **{f"prob_quiebre_{c}": np.where(np.isnan(s0), np.nan, pc[:, j]) for j, c in enumerate(CORTES)},
            **{f"frac_{c}": fr[f"p_{c}"].values for c in (900, 1100, 1300)},
            "dias_cobertura_p90": s0 / np.where(pr.q90.values > 0, pr.q90.values, np.nan),
            "nivel_confianza": conf.reindex(pr.id_sucursal).values, "factores": pr.factores.values})
        pred["retiro_p50"] = np.clip(pred.retiro_p50, pred.retiro_p10, pred.retiro_p90)
        pred.insert(pred.columns.get_loc("dias_cobertura_p90"), "dias_cobertura_p50",
                    s0 / pred.retiro_p50.where(pred.retiro_p50 > 0).values)
    run = {"id_politica_run": rid, "fecha_corte": fecha_corte, "fecha_objetivo": T, "regimen": aux["regimen"],
           "version_modelo": cfg["version_modelo"], "config_hash": cfg.hash, "huella_insumos": json.dumps(huella),
           "parametros": json.dumps(cfg.raw, default=str), "estado": "OK" if n_rech_perfil == 0 and len(rech_c) == 0 else "OK_CON_RECHAZOS",
           "n_perfiles": int(len(perf_ok)), "n_cortes": int(len(corte_ok)), "n_exclusiones": int(len(excl)),
           "n_predicciones": int(len(pred)) if pred is not None else 0, "duracion_seg": round(time.time() - t0, 1),
           "mensaje": "; ".join([f"perfiles rechazados por contrato: {n_rech_perfil}", f"intervalos rechazados: {len(rech_c)}"] + avisos), "vigente": 1, "modo": modo}
    return {"run": run, "perf_ok": perf_ok, "corte_ok": corte_ok, "excl": excl, "pred": pred, "suc": suc}


_PK = {"perf_ok": ["id_sucursal"], "corte_ok": ["id_sucursal", "hora_inicio", "hora_fin"],
       "excl": ["id_sucursal", "motivo"], "pred": ["id_sucursal"]}


def huella_resultado(r: dict) -> str:
    """Hash exacto (bit a bit) de lo que se publicaría, sin marcas de tiempo ni duración."""
    import hashlib
    h = hashlib.sha256()
    for k, pk in _PK.items():
        df = r[k]
        if df is None:
            h.update(b"<none>")
            continue
        df = df.drop(columns=[c for c in ("created_at_utc",) if c in df.columns]).sort_values(pk, kind="mergesort")
        h.update(",".join(df.columns).encode())
        h.update(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    h.update(json.dumps({k: v for k, v in r["run"].items() if k != "duracion_seg"}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def correr_diario(cfg, fecha_corte: int, errores_semilla: pd.DataFrame | None = None, force: bool = False,
                  db_path=None, lista_8am=None, solo_perfil: bool = False, modo: str | None = None) -> dict:
    """db_path=None -> destino real (config destino.sqlite). solo_perfil=True -> carga histórica sin predicciones.

    Cómputo verificado (EV-16): el cálculo completo se hace `modelo.computos_verificacion` veces (2 por defecto), cada
    vez desde la extracción, y solo se publica si dos cómputos coinciden BIT A BIT. Si no coinciden o uno falla, se
    hacen cómputos adicionales (máximo `modelo.computos_max`, 5 por defecto) hasta que dos coincidan; si no, no se publica.
    """
    import gc
    from collections import Counter
    modo = modo or ("PRUEBA" if db_path else ("HISTORICO" if solo_perfil else cfg["destino"]["modo"]))
    destino = db_path or ruta_destino(cfg)
    t0 = time.time()
    k = max(1, int(cfg["modelo"].get("computos_verificacion", 2)))
    max_intentos = max(k, int(cfg["modelo"].get("computos_max", 5)))
    hechos, fallas = [], []
    for intento in range(1, max_intentos + 1):
        try:
            r = _calcular(cfg, fecha_corte, errores_semilla, db_path, lista_8am, solo_perfil, modo)
            hechos.append((r, huella_resultado(r)))
            log.info("cómputo %d: huella del resultado %s", intento, hechos[-1][1][:16])
        except CorridaAbortada:
            raise                                     # problema de datos: es determinista, repetir no ayuda
        except Exception as e:  # noqa: BLE001
            fallas.append(f"cómputo {intento}: {type(e).__name__}: {e}")
            log.error("cómputo %d FALLÓ: %s", intento, e)
        gc.collect()
        conteo = Counter(h for _, h in hechos)
        if conteo and conteo.most_common(1)[0][1] >= min(2, k):
            break
        if intento >= 2 and len(hechos) >= 2 and len(conteo) > 1:
            log.error("Los cómputos NO coinciden (%s): se hace un cómputo adicional", [h[:12] for _, h in hechos])
    conteo = Counter(h for _, h in hechos)
    if not conteo or conteo.most_common(1)[0][1] < min(2, k):
        raise RuntimeError(f"Cómputo no verificado: {len(hechos)} resultados {[h[:12] for _, h in hechos]}, "
                           f"fallas {fallas}. No se publica nada (EV-16).")
    h_ok = conteo.most_common(1)[0][0]
    r = next(x for x, h in hechos if h == h_ok)
    run, perf_ok, corte_ok, excl, pred, suc = (r[x] for x in ("run", "perf_ok", "corte_ok", "excl", "pred", "suc"))
    rid = run["id_politica_run"]
    n_ok = sum(h == h_ok for _, h in hechos)
    run["mensaje"] = f"{run['mensaje']}; cómputo verificado: {n_ok} de {len(hechos) + len(fallas)} coincidentes" + \
        (f" ({len(fallas)} fallidos, {len(hechos) - n_ok} distintos)" if fallas or len(hechos) > n_ok else "")
    run["duracion_seg"] = round(time.time() - t0, 1)
    # 9 persistencia: destino real (bancoDBPlanificacion_minimo.db) o base de pruebas (--db)
    if db_path:
        con = conectar_dev(cfg, db_path)
        estado = escribir(con, run, perf_ok, corte_ok, excl, pred, suc, force=force)
    else:
        if cfg["destino"].get("respaldo_antes_de_primera_escritura", True):
            bak = respaldar(destino, conservar=int(cfg["destino"].get("respaldos_a_conservar", 5)))
            if bak:
                log.info("respaldo del destino: %s", bak.name)
        con = conectar_destino(cfg, destino)
        cat = sucursales_catalogo(con)          # FK: no se toca el catálogo; se excluye lo que no exista en SUCURSAL
        fuera = set(perf_ok.id_sucursal) - cat
        if fuera:
            excl = pd.concat([excl, pd.DataFrame({"id_politica_run": rid, "id_sucursal": sorted(fuera), "motivo": "FK_SUCURSAL",
                                                  "detalle": "no existe en SUCURSAL del destino"})], ignore_index=True)
            perf_ok, corte_ok = perf_ok[~perf_ok.id_sucursal.isin(fuera)], corte_ok[~corte_ok.id_sucursal.isin(fuera)]
            run.update(n_perfiles=int(len(perf_ok)), n_cortes=int(len(corte_ok)), n_exclusiones=int(len(excl)))
        estado = escribir(con, run, perf_ok, corte_ok, excl, pred, None, force=force)
    con.close()
    if not db_path and cfg["destino"]["postgres"].get("habilitado"):
        # Vuelca TODAS las corridas pendientes (pg_sincronizado = 0), no solo la de hoy: si una noche falla
        # Postgres, la siguiente lo recupera. Un fallo aquí no invalida lo ya escrito en SQLite.
        from .postgres_sync import sincronizar
        try:
            run["postgres"] = sincronizar(cfg, commit=True)
        except Exception as e:  # noqa: BLE001
            log.error("Volcado a Postgres FALLÓ (queda pendiente para la siguiente corrida): %s", e)
            run["postgres"] = {"resultado": "FALLÓ", "error": str(e)}
    run["escritura"] = estado
    log.info("corrida %s: %s (%s perfiles, %s exclusiones, %s predicciones) en %.1fs", rid, estado,
             run["n_perfiles"], run["n_exclusiones"], run["n_predicciones"], time.time() - t0)
    return run
