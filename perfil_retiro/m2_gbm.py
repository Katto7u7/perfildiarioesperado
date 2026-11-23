"""M2: gradient boosting global (pérdida cuantil 0.5) sobre log1p(retiro). Candidato pre-registrado (docs/03 §4).

Anti-fuga: cada fila (sucursal, T) usa SOLO observaciones válidas con fecha < T (merge_asof estricto).
Reentrenamiento: primer origen de cada mes, con objetivos ya realizados (T <= origen de reentrenamiento).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .modelo import contexto_b1, variables

CAL = ["am1", "am2", "becas1", "becas2", "impar", "martes", "miercoles", "jueves", "viernes", "quincena",
       "log_monto_am", "stock_am_h2", "stock_am_h8", "stock_becas_h2", "stock_becas_h8", "evento_pe"]


def _asof(grid, feats, by, cols):
    g = grid.sort_values("id_fecha")
    f = feats.sort_values("id_fecha")[["id_fecha"] + by + cols]
    m = pd.merge_asof(g, f, on="id_fecha", by=by, allow_exact_matches=False, direction="backward")
    return m.set_index(g.index).loc[grid.index, cols]


def construir_features(panel: pd.DataFrame, letras: pd.Series) -> pd.DataFrame:
    """Una fila por (sucursal, día operativo) con variables de calendario del día y estadísticos previos."""
    p = panel[panel.id_fecha >= 20250602].copy()
    X = variables(p)
    p["ctx"] = contexto_b1(X).values
    p["llave"] = p.id_fecha.map(letras).fillna("SIN")
    for c in CAL:
        p[c] = X[c].values
    p["dhm"] = p.dia_habil_mes
    v = p[p.valido].sort_values(["id_sucursal", "id_fecha"]).copy()
    v["ly"] = np.log1p(v.retiro)
    g = v.groupby("id_sucursal").ly
    v["f_l1"] = v.ly
    v["f_m5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
    v["f_md20"] = g.transform(lambda s: s.rolling(20, min_periods=3).median())
    v["f_sd20"] = g.transform(lambda s: s.rolling(20, min_periods=3).std())
    for r in ("IMPAR", "PAR"):
        z = v.ly.where(v.regimen == r)
        v[f"f_reg_{r}"] = z.groupby(v.id_sucursal).transform(lambda s: s.rolling(120, min_periods=1).mean().ffill())
    v["f_ctx8"] = v.groupby(["id_sucursal", "ctx"]).ly.transform(lambda s: s.rolling(8, min_periods=1).median())
    v["f_let3"] = v.groupby(["id_sucursal", "llave"]).ly.transform(lambda s: s.rolling(3, min_periods=1).median())
    base = ["f_l1", "f_m5", "f_md20", "f_sd20", "f_reg_IMPAR", "f_reg_PAR"]
    out = p[["id_sucursal", "id_fecha", "regimen", "valido", "retiro", "ctx", "llave", "dhm"] + CAL].copy()
    out[base] = _asof(out, v, ["id_sucursal"], base)
    out["f_ctx8"] = _asof(out, v, ["id_sucursal", "ctx"], ["f_ctx8"])
    out["f_let3"] = _asof(out, v, ["id_sucursal", "llave"], ["f_let3"])
    out.loc[out.llave == "SIN", "f_let3"] = np.nan
    out["f_reg_obj"] = np.where(out.regimen == "IMPAR", out.f_reg_IMPAR, out.f_reg_PAR)
    out["f_reg_otro"] = np.where(out.regimen == "IMPAR", out.f_reg_PAR, out.f_reg_IMPAR)
    return out


FEATS = CAL + ["dhm", "f_l1", "f_m5", "f_md20", "f_sd20", "f_reg_obj", "f_reg_otro", "f_ctx8", "f_let3"]


def predecir_backtest(fe: pd.DataFrame, pares: pd.DataFrame, ventana: int = 160, seed: int = 0, log=print) -> pd.Series:
    """pares: DataFrame con (origen, id_fecha, id_sucursal). Devuelve pronóstico p50 (pesos) alineado a `pares`."""
    pares = pares.copy()
    pares["mes"] = pares.origen // 100
    fe = fe.set_index(["id_fecha", "id_sucursal"])
    res = pd.Series(np.nan, index=pares.index)
    for mes, grp in pares.groupby("mes"):
        o0 = int(grp.origen.min())
        tr = fe[(fe.index.get_level_values(0) <= o0) & fe.valido].reset_index()
        fechas = np.sort(tr.id_fecha.unique())[-ventana:]
        tr = tr[tr.id_fecha.isin(fechas)].dropna(subset=["f_md20"])
        m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.08,
                                          max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=seed)
        m.fit(tr[FEATS].astype(float), np.log1p(tr.retiro))
        q = fe.reindex(pd.MultiIndex.from_arrays([grp.id_fecha, grp.id_sucursal]))
        res.loc[grp.index] = np.expm1(m.predict(q[FEATS].astype(float))).clip(min=0)
        log(f"M2 mes {mes}: entrenado hasta {o0} con {len(tr)} filas; {len(grp)} pronósticos")
    return res
