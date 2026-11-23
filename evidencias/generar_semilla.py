"""Regenera out/semilla_errores.parquet: errores fuera de muestra del predictor final (ENS_ALL) en el backtest
calibrado de desarrollo + holdout (origenes 2025-12 a 2026-11-13). Es la semilla con la que arranca la calibración
conformal en producción; cada noche los errores reales de la operación sustituyen a la semilla en sus fechas.

Requiere out/backtest/cal_devhold_ens_all.parquet (salida de evidencias/ev12_torneo.py sobre el backtest dev+holdout).
Uso:  python evidencias/generar_semilla.py      -> imprime forma, rango de fechas y sha256 del archivo
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
b = pd.read_parquet(RAIZ / "out" / "backtest" / "cal_devhold_ens_all.parquet")
e = b[(b.estado_real == "OK") & b.real.notna() & b.ens_all.notna()].copy()
sem = pd.DataFrame({"id_sucursal": e.id_sucursal, "estrato": e.estrato, "id_fecha": e.id_fecha,
                    "err": np.log1p(e.real) - np.log1p(e.ens_all), "real": e.real, "pt": e.ens_all, "real_ok": e.real,
                    "abs_err": (e.ens_all - e.real).abs(),
                    "dentro": ((e.real >= e.c_q10) & (e.real <= e.c_q90)).astype(float)})
out = RAIZ / "out" / "semilla_errores.parquet"
sem.to_parquet(out, index=False)
print("semilla", sem.shape, sem.id_fecha.min(), sem.id_fecha.max(), hashlib.sha256(out.read_bytes()).hexdigest())
