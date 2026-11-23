"""EV-15 Riesgo de saldo con las definiciones del usuario (2026-11-18), sin dotación adicional.

  P(crítico al cierre) = P(S0 + dep_1700 − R < P90)          -> la sucursal cierra con < 1 día de demanda alta
  P(quiebre intradía)  = max_c P(S0 + dep_c − f_c·R < (1−f_c)·P90)  (c = 9/11/13) y P(S0 + dep_1700 − R < 0)
Evaluación: desarrollo (confirmatoria para la definición nueva) y holdout (POST HOC: la definición cambió tras verlo).
Realizado: saldos por corte del día objetivo SIN las dotaciones del día (misma premisa que el pronóstico).
"""
import numpy as np
import pandas as pd

from comun import RAIZ, guardar_json, panel_completo, salida
from perfil_retiro.backtest import cdf_desde_cuantiles
from perfil_retiro.calibracion import QCOLS
from perfil_retiro.evaluacion import calibracion_prob

out = salida("EV-15")
b = pd.read_parquet(RAIZ / "out" / "backtest" / "cal_devhold_ens_all.parquet")
b = b[(b.estado_real == "OK") & (b.c_fuente == "CONFORMAL") & b.saldo_ini.notna()]
p = panel_completo()
cols = ["id_fecha", "id_sucursal", "sdo_inicial", "forma_valida"] + [f"{a}_{c}" for a in ("depositos", "dotaciones", "retiros")
                                                                      for c in (900, 1100, 1300, 1700)]
b = b.merge(p[cols].rename(columns={"forma_valida": "fv_T"}), on=["id_fecha", "id_sucursal"], how="left")
qm = b[["c_" + q for q in QCOLS]].values
U = b.c_q90.values
S0 = b.saldo_ini.values
pq = []
for c in (900, 1100, 1300):
    f = np.maximum(b[f"f_{c}"].values, 1e-3)
    pq.append(1 - cdf_desde_cuantiles(qm, (S0 + b[f"dep_{c}"].values - (1 - f) * U) / f))
pq.append(1 - cdf_desde_cuantiles(qm, S0 + b.dep_1700.values))
pq = np.column_stack(pq)
b["p_quiebre"] = pq.max(axis=1)
b["p_critico_cierre"] = 1 - cdf_desde_cuantiles(qm, S0 + b.dep_1700.values - U)
# realizado (sin dotación del día)
s = {c: b.sdo_inicial + b[f"depositos_{c}"] - b[f"dotaciones_{c}"] - b[f"retiros_{c}"] for c in (900, 1100, 1300, 1700)}
rq = np.zeros(len(b), bool)
for c in (900, 1100, 1300):
    rq |= (s[c] < (1 - b[f"f_{c}"]) * U).fillna(False).values
rq |= (s[1700] < 0).fillna(False).values
ok = b[[f"retiros_{c}" for c in (900, 1100, 1300, 1700)]].notna().all(axis=1) & b.sdo_inicial.notna()
b["real_quiebre"] = np.where(ok, rq.astype(float), np.nan)
b["real_critico_cierre"] = np.where(ok, (s[1700] < U).astype(float), np.nan)

res = {}
for per, (a, z) in {"desarrollo": (0, 20260930), "holdout_POST_HOC": (20261001, 29991231)}.items():
    x = b[b.id_fecha.between(a, z)]
    for ev, pcol, rcol in (("critico_cierre", "p_critico_cierre", "real_critico_cierre"), ("quiebre_intradia", "p_quiebre", "real_quiebre")):
        for sub, xx in (("todos", x), ("sin_traspaso", x[x.fv_T.astype("boolean").fillna(False).astype(bool)])):
            t, m = calibracion_prob(xx[pcol].values, xx[rcol].values)
            al = xx[pcol] >= 0.5
            tp = (al & (xx[rcol] == 1)).sum()
            m.update({"precision@0.5": float(tp / max(al.sum(), 1)), "recall@0.5": float(tp / max((xx[rcol] == 1).sum(), 1)),
                      "max_error_decil_pp": float(t.error_abs_pp[t.n >= 200].max())})
            t.to_csv(out / f"calibracion_{ev}_{per}_{sub}.csv")
            res[f"{per}|{ev}|{sub}"] = m
guardar_json("EV-15", res)
for k, v in res.items():
    print(k, {a: round(c, 4) for a, c in v.items()})
