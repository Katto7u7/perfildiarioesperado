"""M2 sobre los mismos pares (origen, sucursal) de un backtest de M1. Uso: python correr_m2.py ETIQUETA_BT"""
import sys, time
import pandas as pd
from comun import RAIZ, fuente, panel_completo
from perfil_retiro.backtest import letras_por_fecha
from perfil_retiro.m2_gbm import construir_features, predecir_backtest

cache = RAIZ / "out" / "cache" / "features_m2.parquet"
if cache.exists():
    fe = pd.read_parquet(cache)
else:
    t = time.time()
    fe = construir_features(panel_completo(), letras_por_fecha(fuente("dispersion")))
    fe.to_parquet(cache, index=False)
    print(f"features {fe.shape} en {time.time()-t:.0f}s")
if len(sys.argv) > 1:
    f = RAIZ / "out" / "backtest" / f"bt_{sys.argv[1]}.parquet"
    bt = pd.read_parquet(f)
    bt["m2"] = predecir_backtest(fe, bt[["origen", "id_fecha", "id_sucursal"]]).values
    bt.to_parquet(f, index=False)
    print("M2 agregado a", f.name)
