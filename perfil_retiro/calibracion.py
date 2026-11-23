"""Cuantiles por conformal CAUSAL sobre errores fuera de muestra del predictor final.

Para el origen D solo se usan errores de pronósticos cuyo día objetivo ya ocurrió (id_fecha <= D).
Es el mismo procedimiento en producción: errores guardados de corridas anteriores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import CORTES, cdf_desde_cuantiles
from .modelo import QUANTS

QCOLS = [f"q{int(a*100):02d}" for a in QUANTS]


def conformal_causal(bt: pd.DataFrame, punto: str, n_suc: int = 120, n0: float = 10.0,
                     min_global: int = 2000, media: str = "razon", n_razon_dias: int = 40) -> pd.DataFrame:
    """Agrega columnas c_q05..c_q99, c_media para el predictor `punto`. bt: salida de backtest.correr."""
    bt = bt.sort_values(["origen", "id_sucursal"]).copy()
    bt["lp"] = np.log1p(bt[punto].clip(lower=0))
    okr = (bt.estado_real == "OK") & bt.real.notna() & bt[punto].notna()
    bt["err"] = np.where(okr, np.log1p(bt.real.clip(lower=0)) - bt.lp, np.nan)
    bt["real_ok"] = np.where(okr, bt.real, np.nan)
    bt["reg"] = np.where(((bt.id_fecha // 100) % 100) % 2 == 1, "IMPAR", "PAR")
    errs = bt[okr][["id_fecha", "id_sucursal", "estrato", "err", "real_ok", punto, "reg"]].rename(columns={punto: "pt"})
    out = []
    for D, g in bt.groupby("origen", sort=True):
        e = errs[errs.id_fecha <= D]
        g = g.copy()
        if len(e) < min_global:                       # calentamiento: cuantiles del modelo reescalados al punto
            for qc in QCOLS:
                g["c_" + qc] = g[qc] * (g[punto] / g.p50.replace(0, np.nan)).fillna(1.0)
            g["c_media"] = g.media * (g[punto] / g.p50.replace(0, np.nan)).fillna(1.0)
            g["c_fuente"] = "MODELO"
            out.append(g)
            continue
        eg = e.groupby("estrato").tail(60000)
        qg = eg.groupby("estrato").err.quantile(QUANTS).unstack()
        smg = np.exp(eg.err.clip(upper=eg.err.quantile(0.999))).groupby(eg.estrato).mean()
        es = e.sort_values("id_fecha").groupby(["id_sucursal", "estrato"]).tail(n_suc)
        qs = es.groupby(["id_sucursal", "estrato"]).err.quantile(QUANTS).unstack()
        ns = es.groupby(["id_sucursal", "estrato"]).size()
        sms = np.exp(es.err.clip(upper=3)).groupby([es.id_sucursal, es.estrato]).mean()
        idx = pd.MultiIndex.from_arrays([g.id_sucursal, g.estrato])
        w = (ns.reindex(idx).fillna(0).values / (ns.reindex(idx).fillna(0).values + n0))[:, None]
        Qs, Qg = qs.reindex(idx).values, qg.reindex(g.estrato).values
        Q = np.where(np.isnan(Qs), Qg, w * Qs + (1 - w) * Qg)
        Q = np.maximum.accumulate(Q, axis=1)
        for i, qc in enumerate(QCOLS):
            g["c_" + qc] = np.expm1(g.lp.values + Q[:, i]).clip(min=0)
        sm = np.where(np.isnan(sms.reindex(idx).values), smg.reindex(g.estrato).values,
                      w[:, 0] * sms.reindex(idx).values + (1 - w[:, 0]) * smg.reindex(g.estrato).values)
        if media in ("razon", "razon_regimen"):
            # estimador de razón (insesgado en la suma): Σ real / Σ pronóstico en los últimos días ya realizados.
            # v1.0.1 (post-holdout, EV-13): con 'razon_regimen' solo se usan días del MISMO régimen que el objetivo.
            reg_obj = g.reg.iloc[0]
            base = e[e.reg == reg_obj] if media == "razon_regimen" and (e.reg == reg_obj).sum() > min_global else e
            fechas = np.sort(base.id_fecha.unique())[-n_razon_dias:]
            er = base[base.id_fecha.isin(fechas)]
            Rg = er.groupby("estrato").real_ok.sum() / er.groupby("estrato").pt.sum()
            Rs_num = es.groupby(["id_sucursal", "estrato"]).real_ok.sum()
            Rs_den = es.groupby(["id_sucursal", "estrato"]).pt.sum()
            Rs = (Rs_num / Rs_den.where(Rs_den > 0)).clip(0.2, 5)
            rs, rg = Rs.reindex(idx).values, Rg.reindex(g.estrato).values
            wr = w[:, 0] * 0.5                                  # la razón propia es ruidosa: contracción adicional
            R = np.where(np.isnan(rs), rg, wr * rs + (1 - wr) * rg)
            g["c_media"] = (g[punto].values * R).clip(min=0)
        else:
            g["c_media"] = (np.exp(g.lp.values) * sm - 1).clip(min=0)
        g["c_fuente"] = "CONFORMAL"
        out.append(g)
    return pd.concat(out, ignore_index=True)


def riesgo(bt: pd.DataFrame, factor: float = 1.0, pref: str = "c_") -> pd.DataFrame:
    """Recalcula P(crítico) por corte con los cuantiles calibrados; evento real con el mismo umbral."""
    bt = bt.copy()
    qmat = bt[[pref + q for q in QCOLS]].values
    U = factor * bt[pref + "q90"].values
    pcs = []
    for c in CORTES:
        f = bt[f"f_{c}"].values
        x = (bt.saldo_ini.values + bt[f"dep_{c}"].values - U) / np.maximum(f, 1e-3)
        pcs.append(1 - cdf_desde_cuantiles(qmat, x))
    pc = np.column_stack(pcs)
    bt["r_umbral"] = U
    bt["r_p_critico"] = np.where(np.isnan(bt.saldo_ini.values), np.nan, pc.max(axis=1))
    for j, c in enumerate(CORTES):
        bt[f"r_p_{c}"] = pc[:, j]
    bt["r_corte_presion"] = np.where(pc.max(axis=1) >= 0.5, np.array(CORTES)[np.argmax(pc >= 0.5, axis=1)], 0)
    sm = bt.saldo_min_sin_dot_real.values
    bt["r_evento_real"] = np.where(np.isnan(sm), np.nan, (sm < U).astype(float))
    return bt
