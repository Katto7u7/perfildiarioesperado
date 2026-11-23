"""Perfil de comportamiento por sucursal -> PERFIL_RETIRO_SUCURSAL y PERFIL_RETIRO_SUCURSAL_CORTE.

Definiciones: docs/02 §3 y docs/04_diccionario.md. Ventana: días válidos del régimen vigente (EV-09), últimos N.
Intradía en hora LOCAL de la sucursal con desfase efectivo de corte δ (EV-11).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modelo import contexto_b1, variables

CORTES = [900, 1100, 1300, 1700]


# --------------------------------------------------------------------------- horario
def horario_verano_eua(fecha: int) -> bool:
    """Horario de verano de EUA (lo sigue la franja fronteriza de Baja California): del 2.º domingo de marzo
    al 1.er domingo de noviembre (el cambio es a las 2:00; a la hora de operación de la sucursal ya aplica)."""
    import datetime as _dt
    y, m, d = fecha // 10000, (fecha // 100) % 100, fecha % 100
    f = _dt.date(y, m, d)
    mar1, nov1 = _dt.date(y, 3, 1), _dt.date(y, 11, 1)
    inicio = mar1 + _dt.timedelta(days=(6 - mar1.weekday()) % 7 + 7)      # 2.º domingo de marzo
    fin = nov1 + _dt.timedelta(days=(6 - nov1.weekday()) % 7)             # 1.er domingo de noviembre
    return inicio <= f < fin


def horario_sucursal(suc: pd.DataFrame, cfg, p900_mediana: pd.Series | None, fecha: int,
                     lista_8am: set | None = None) -> pd.DataFrame:
    """offset de zona (h vs centro), apertura local (HHMM) y fuente de la apertura por sucursal."""
    h = cfg["horario"]
    off = suc.entidad.map(h["offset_entidad"]).fillna(0).astype(int)
    dst = horario_verano_eua(fecha)
    off = np.where((suc.entidad == "BAJA CALIFORNIA") & dst, -1, off)
    out = pd.DataFrame({"id_sucursal": suc.id_sucursal.values, "offset": off})
    out["apertura_local"] = h["apertura_local_hhmm"]
    out["fuente_apertura"] = "DECLARADA_9AM"
    if lista_8am:
        es8 = out.id_sucursal.isin(lista_8am)
        out.loc[es8, ["apertura_local", "fuente_apertura"]] = [h["apertura_transaccional_hhmm"], "LISTA_8AM"]
    elif p900_mediana is not None:
        es8 = (out.offset == 0) & out.id_sucursal.map(p900_mediana).gt(h["umbral_p900_apertura_8"]).fillna(False)
        out.loc[es8, ["apertura_local", "fuente_apertura"]] = [h["apertura_transaccional_hhmm"], "INFERIDA_8AM"]
    return out.set_index("id_sucursal")


def tiempos_corte(hor: pd.DataFrame, delta: float) -> pd.DataFrame:
    """τ_c = horas de operación transcurridas al corte c (etiqueta en hora centro)."""
    ap = hor.apertura_local // 100 + (hor.apertura_local % 100) / 60
    t = pd.DataFrame({c: (c // 100 + (c % 100) / 60) - (ap - hor.offset) + delta for c in CORTES}, index=hor.index)
    return t


def _hhmm(horas: float) -> int:
    h = int(np.floor(horas + 1e-9))
    m = int(round((horas - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return h * 100 + m


# --------------------------------------------------------------------------- perfil
def calcular_perfil(panel: pd.DataFrame, cal: pd.DataFrame, suc: pd.DataFrame, cfg, fecha_corte: int,
                    wape_oos: pd.Series | None = None, lista_8am: set | None = None):
    """Devuelve (perfil, corte, exclusiones, auxiliares). panel: ya acotado a id_fecha <= fecha_corte."""
    P = cfg["perfil"]
    N, nmin, fac = int(P["ventana_dias_habiles_regimen"]), int(P["min_dias_validos"]), float(P["factor_umbral_critico"])
    ops = cal[cal.operativo].id_fecha.sort_values()
    T = int(ops[ops > fecha_corte].iloc[0])                      # día objetivo -> régimen vigente
    reg = cal.set_index("id_fecha").regimen.loc[T]
    p = panel[(panel.id_fecha <= fecha_corte) & (panel.id_fecha >= int(cfg["historia"]["inicio_util"]))].copy()
    v = p[p.valido].sort_values(["id_sucursal", "id_fecha"])

    # umbrales causales: mediana/P90 del mismo régimen en los últimos 60 días válidos ANTES de cada día
    vr = v[v.regimen == reg].copy()
    g = vr.groupby("id_sucursal").retiro
    vr["p50_prev"] = g.transform(lambda s: s.rolling(60, min_periods=20).median().shift(1))
    vr["p90_prev"] = g.transform(lambda s: s.rolling(60, min_periods=20).quantile(0.9).shift(1))
    vr["p50_hoy"] = g.transform(lambda s: s.rolling(60, min_periods=20).median())   # conocido al cierre de d
    vr["p90_hoy"] = g.transform(lambda s: s.rolling(60, min_periods=20).quantile(0.9))
    w = vr.groupby("id_sucursal").tail(N)

    n = w.groupby("id_sucursal").size()
    elegibles = n[n >= nmin].index
    excl = pd.DataFrame({"id_sucursal": n[n < nmin].index, "motivo": "DIAS_VALIDOS_INSUFICIENTES",
                         "detalle": [f"{int(k)} días válidos del régimen {reg} (mínimo {nmin})" for k in n[n < nmin]]})
    sin_dem = w[w.id_sucursal.isin(elegibles)].groupby("id_sucursal").retiro.sum()
    sin_dem = sin_dem[sin_dem <= 0].index
    if len(sin_dem):
        excl = pd.concat([excl, pd.DataFrame({"id_sucursal": sin_dem, "motivo": "SIN_DEMANDA",
                                              "detalle": "retiro = 0 en todos los días válidos de la ventana"})])
        elegibles = elegibles.difference(sin_dem)
    w = w[w.id_sucursal.isin(elegibles)].copy()
    G = w.groupby("id_sucursal")
    r = pd.DataFrame(index=elegibles)
    r["dias_observados"] = n.loc[elegibles]
    r["retiro_promedio_diario"] = G.retiro.mean()
    r["retiro_p90_diario"] = G.retiro.quantile(0.90)
    r["retiro_p95_diario"] = G.retiro.quantile(0.95)
    r["retiro_max_diario"] = G.retiro.max()
    r["volatilidad_retiro_diario"] = G.retiro.std(ddof=1)

    # saldos y cobertura
    ws = w[w.saldo_valido & w.p50_hoy.gt(0)].copy()
    cap = float(P.get("tope_cobertura_dias", 30))
    ws["cob50"] = (ws.saldo_cierre / ws.p50_hoy).clip(upper=cap)
    ws["cob90"] = (ws.saldo_cierre / ws.p90_hoy.where(ws.p90_hoy > 0)).clip(upper=cap)
    r["saldo_promedio_cierre"] = ws.groupby("id_sucursal").saldo_cierre.mean()
    r["dias_cobertura_promedio"] = ws.groupby("id_sucursal").cob50.mean()
    r["dias_cobertura_p90"] = ws.groupby("id_sucursal").cob90.mean()
    # Definiciones (decisión del usuario 2026-11-18):
    #  crítico (cierre)  : saldo de cierre < P90 de la siguiente jornada (conocido al cierre de d) -> cobertura < 1 día
    #  quiebre intradía  : saldo en un corte < P90 de la demanda RESTANTE del día ((1 - f_c) x P90 previo);
    #                      en el corte de 17:00 equivale a saldo < 0 (desabasto)
    #  día en riesgo     : crítico o quiebre -> base del score de riesgo
    wc = w[w.saldo_valido & w.p90_prev.notna() & w.p90_hoy.notna()].copy()
    fmed = w[w.forma_valida].groupby("id_sucursal")[["p_900", "p_1100", "p_1300"]].median()
    fm = fmed.reindex(wc.id_sucursal).fillna(fmed.median()).values
    q = np.zeros(len(wc), bool)
    for j, c in enumerate((900, 1100, 1300)):
        q |= (wc[f"sdo_final_{c}"].values < (1 - fm[:, j]) * fac * wc.p90_prev.values)
    q |= wc.sdo_final_1700.values < 0
    wc["quiebre"] = q
    wc["critico_cierre"] = wc.saldo_cierre < fac * wc.p90_hoy
    wc["riesgo"] = wc.quiebre | wc.critico_cierre
    g_ = wc.groupby("id_sucursal")
    r["pct_dias_saldo_critico"] = g_.critico_cierre.mean()
    r["pct_quiebre_intradia"] = g_.quiebre.mean()
    k_crit, n_crit = g_.riesgo.sum(), g_.riesgo.size()
    m_glob = float(k_crit.sum() / max(n_crit.sum(), 1))
    a = float(P.get("fuerza_previa_riesgo", 20))
    r["score_riesgo_desabasto"] = (k_crit + a * m_glob) / (n_crit + a)

    # intradía (hora local, δ)
    p9 = p[p.forma_valida].groupby("id_sucursal").p_900.median()
    hor = horario_sucursal(suc[suc.id_sucursal.isin(elegibles)], cfg, p9, fecha_corte, lista_8am)
    tau = tiempos_corte(hor, float(cfg["horario"]["desfase_corte_horas"]))
    wf = w[w.forma_valida].copy()
    sh = np.column_stack([wf.p_900, wf.p_1100 - wf.p_900, wf.p_1300 - wf.p_1100, 1 - wf.p_1300])
    tv = tau.reindex(wf.id_sucursal).values
    t0 = np.clip(tv, 0, None)
    dur = np.column_stack([t0[:, 0], t0[:, 1] - t0[:, 0], t0[:, 2] - t0[:, 1], t0[:, 3] - t0[:, 2]])
    activo = dur > 0.05                                         # intervalo existe (abierto) para esa zona
    vel = np.where(activo, sh * wf.retiro.values[:, None] / np.where(activo, dur, 1), np.nan)
    mid = np.column_stack([t0[:, 0] / 2, (t0[:, 0] + t0[:, 1]) / 2, (t0[:, 1] + t0[:, 2]) / 2, (t0[:, 2] + t0[:, 3]) / 2])
    acc = np.abs(np.diff(vel, axis=1)) / np.where(np.diff(mid, axis=1) > 0, np.diff(mid, axis=1), np.nan)
    tf = dur / np.where(dur.sum(axis=1, keepdims=True) > 0, dur.sum(axis=1, keepdims=True), np.nan)
    conc = np.nansum(np.where(activo, sh ** 2 / np.where(tf > 0, tf, np.nan), 0), axis=1) - 1
    agres = wf.p_1100.values / np.where(t0[:, 1] > 0, t0[:, 1] / t0[:, 3], np.nan)
    wf["agres"], wf["conc"] = agres, conc
    vdf = pd.DataFrame(vel, index=wf.id_sucursal.values, columns=CORTES)
    adf = pd.DataFrame(acc, index=wf.id_sucursal.values)
    # promedio = ritmo medio de la jornada (retiro del día / horas de operación), promediado en días;
    # p90 = p90 entre días de la velocidad del tramo pico. (Definición 2026-11-20: la versión que mezclaba
    # intervalos de distinta duración en un solo pool violaba el CHECK p90 >= promedio en ~2% de sucursales.)
    horas_dia = np.where(activo, dur, 0).sum(axis=1)
    vel_dia = wf.retiro.values / np.where(horas_dia > 0, horas_dia, np.nan)
    vel_pico = np.nanmax(vel, axis=1)
    r["velocidad_consumo_promedio"] = pd.Series(vel_dia, index=wf.id_sucursal.values).groupby(level=0).mean()
    r["velocidad_consumo_p90"] = pd.Series(vel_pico, index=wf.id_sucursal.values).groupby(level=0).quantile(0.9)
    r["aceleracion_abs_p90"] = adf.stack().groupby(level=0).quantile(0.9)
    r["indice_agresividad_consumo"] = wf.groupby("id_sucursal").agres.median()
    r["indice_concentracion_intradia"] = wf.groupby("id_sucursal").conc.median().clip(lower=0)
    vmed = vdf.groupby(level=0).median()
    shdf = pd.DataFrame(sh, index=wf.id_sucursal.values, columns=CORTES).groupby(level=0).median()
    ipk = vmed.idxmax(axis=1)
    r["dias_forma_valida"] = wf.groupby("id_sucursal").size()
    filas_corte = []
    ap_h = hor.apertura_local // 100 + (hor.apertura_local % 100) / 60
    for s_ in r.index:
        if s_ not in vmed.index or pd.isna(ipk.get(s_)):
            continue
        O = float(ap_h.get(s_, 9.0))
        tt = np.clip(tau.loc[s_].values, 0, None)              # horas de operación al corte (efectivas)
        ini = [_hhmm(O)] + [_hhmm(O + tt[j - 1]) for j in range(1, 4)]
        fin = [_hhmm(O + tt[j]) for j in range(4)]
        c_pk = int(ipk[s_])
        j = CORTES.index(c_pk)
        r.loc[s_, "hora_pico_consumo"] = ini[j]
        r.loc[s_, "intervalo_pico"] = f"{ini[j]:04d}-{fin[j]:04d}"
        r.loc[s_, "participacion_intervalo_pico"] = shdf.loc[s_, c_pk]
        x = wf[wf.id_sucursal == s_]
        for jj, c in enumerate(CORTES):
            if fin[jj] <= ini[jj]:                              # intervalo inexistente en esa zona/apertura
                continue
            frac = x.p_900 if jj == 0 else (x[f"p_{CORTES[jj]}"] if jj < 3 else 1.0) - x[f"p_{CORTES[jj-1]}"]
            monto = frac * x.retiro
            filas_corte.append({"id_sucursal": int(s_), "hora_inicio": ini[jj], "hora_fin": fin[jj],
                                "dias_observados_intervalo": int(len(x)), "retiro_promedio_intervalo": float(monto.mean()),
                                "retiro_p90_intervalo": float(monto.quantile(0.9)), "retiro_p95_intervalo": float(monto.quantile(0.95)),
                                "participacion_dia": float(frac.median())})
    corte = pd.DataFrame(filas_corte)

    # consumo extremo y volatilidad residual (vs contexto de calendario)
    ctx = contexto_b1(variables(w)).str.split("_").str[0].values
    w["ly"], w["ctx"] = np.log1p(w.retiro), ctx
    med = w.groupby(["id_sucursal", "ctx"]).ly.transform("median")
    nctx = w.groupby(["id_sucursal", "ctx"]).ly.transform("size")
    med = np.where(nctx >= 5, med, w.groupby("id_sucursal").ly.transform("median"))
    w["res"] = w.ly - med
    mad = w.groupby("id_sucursal").res.transform(lambda s: 1.4826 * (s - s.median()).abs().median()) + 1e-6
    r["score_volatilidad"] = w.groupby("id_sucursal").res.apply(lambda s: 1.4826 * (s - s.median()).abs().median())
    r["pct_dias_consumo_extremo"] = ((w.res - w.groupby("id_sucursal").res.transform("median")) / mad > 3).groupby(w.id_sucursal).mean()
    vv = v.groupby("id_sucursal").tail(250)
    cx = contexto_b1(variables(vv)).str.split("_").str[0]
    m_am1 = vv.retiro[cx.values == "AM1"].groupby(vv.id_sucursal[cx.values == "AM1"]).median()
    sin = cx.values == "SIN"
    m_sin = vv.retiro[sin].groupby(vv.id_sucursal[sin]).agg(["median", "size"])
    m_sin = m_sin["median"].where(m_sin["size"] >= 5)
    r["intensidad_dispersion"] = (m_am1 / m_sin.where(m_sin > 0)).reindex(r.index)

    # predictibilidad: evidencia fuera de muestra (no un puntaje inventado)
    if wape_oos is not None:
        r["score_predictibilidad"] = (1 - wape_oos.reindex(r.index)).clip(lower=0)
    else:
        r["score_predictibilidad"] = np.nan

    # tipo_perfil (reglas por prioridad, umbrales absolutos de configuración)
    t = cfg["tipo_perfil"]
    cond = [r.score_riesgo_desabasto >= t["alto_riesgo_score"],
            r.pct_quiebre_intradia >= t["quiebre_intradia_pct"],
            (r.dias_cobertura_promedio >= t["sobredotada_cobertura_dias"]) & (r.score_riesgo_desabasto < t["sobredotada_riesgo_max"]),
            r.intensidad_dispersion >= t["dispersion_intensidad"],
            r.score_volatilidad >= t["volatil_score"]]
    r["tipo_perfil"] = np.select(cond, ["ALTO_RIESGO", "QUIEBRE_INTRADIA", "SOBREDOTADA", "DISPERSION", "VOLATIL"], "ESTABLE")
    r["regimen_perfil"] = reg
    r = r.join(hor[["offset", "apertura_local", "fuente_apertura"]])
    return r.reset_index().rename(columns={"index": "id_sucursal"}), corte, excl, {"T": T, "regimen": reg}
