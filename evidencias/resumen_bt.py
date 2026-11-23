"""Resumen rápido de un archivo de backtest: python resumen_bt.py ETIQUETA"""
import sys
import numpy as np
import pandas as pd
from comun import RAIZ
r = pd.read_parquet(RAIZ / "out" / "backtest" / f"bt_{sys.argv[1]}.parquet")
r = r[r.estado_real == "OK"]
print(len(r), "filas", r.origen.nunique(), "orígenes")
for m in ["p50", "b0", "b1", "b2", "nb", "fenix"]:
    if m not in r:
        continue
    x = r.dropna(subset=[m])
    if len(x) == 0:
        continue
    print(f"{m:6s} n={len(x):7d} WAPE={(x[m]-x.real).abs().sum()/x.real.sum():.3f} sesgo={(x[m]-x.real).sum()/x.real.sum():+.3f}")
print("cobertura P10-P90:", round(((r.real >= r.q10) & (r.real <= r.q90)).mean(), 3), " real<=P95:", round((r.real <= r.q95).mean(), 3))
r["mes"] = r.id_fecha // 100
g = r.groupby("mes").apply(lambda x: pd.Series({"wape_m1": (x.p50 - x.real).abs().sum() / x.real.sum(),
                                                 "wape_b1": (x.b1 - x.real).abs().sum() / x.real.sum(),
                                                 "sesgo_m1": (x.p50 - x.real).sum() / x.real.sum(),
                                                 "cov": ((x.real >= x.q10) & (x.real <= x.q90)).mean()}), include_groups=False)
print(g.round(3).to_string())
