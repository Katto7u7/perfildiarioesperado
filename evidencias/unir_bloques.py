"""Une bloques mensuales del backtest: python unir_bloques.py PREFIJO SALIDA"""
import sys, glob
import pandas as pd
from comun import RAIZ
d = RAIZ / "out" / "backtest"
fs = sorted(glob.glob(str(d / f"bt_{sys.argv[1]}_2*.parquet")))
r = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(["origen", "id_sucursal"])
r.to_parquet(d / f"bt_{sys.argv[2]}.parquet", index=False)
print(len(fs), "bloques ->", len(r), "filas,", r.origen.nunique(), "orígenes,", r.id_fecha.min(), "-", r.id_fecha.max())
