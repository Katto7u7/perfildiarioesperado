"""Ejecuta el backtest del protocolo. Uso: python correr_backtest.py DESDE HASTA ETIQUETA [vida_media] [n0]"""
import sqlite3
import sys
import time

import pandas as pd

from comun import CFG, RAIZ, calendario_op, fuente, panel_completo
from perfil_retiro.backtest import correr

desde, hasta, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
vm = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
n0 = float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
reg = (sys.argv[6] != "0") if len(sys.argv) > 6 else True
cada = int(sys.argv[7]) if len(sys.argv) > 7 else 1
p = panel_completo()
p = p[p.id_fecha >= int(CFG["historia"]["inicio_util"])]
cal = calendario_op()
con = sqlite3.connect("file:" + str(CFG.ruta("db_operativo")).replace("\\", "/") + "?mode=ro", uri=True)
pe = pd.read_sql("select evento_codigo, id_sucursal, id_fecha_solicitud, id_fecha_evento from PROGRAMAS_EMERGENTES", con)
con.close()
fx = fuente("prediccion_ds")
t = time.time()
r = correr(p, cal, fuente("dispersion"), pe, fx, desde, hasta, vida_media=vm, n0=n0, regimen=reg, cada=cada)
out = RAIZ / "out" / "backtest"
out.mkdir(parents=True, exist_ok=True)
r.to_parquet(out / f"bt_{tag}.parquet", index=False)
print(f"listo {len(r)} filas, {r.origen.nunique()} orígenes, {time.time()-t:.0f}s -> bt_{tag}.parquet")
