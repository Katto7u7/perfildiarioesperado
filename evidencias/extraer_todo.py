"""Extrae a caché (parquet) todas las fuentes con historia completa hasta la última fecha disponible."""
import logging, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from perfil_retiro.config import cargar_config
from perfil_retiro.sources import CONSULTAS, extraer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = cargar_config()
HASTA = 29991231
for n in CONSULTAS:
    t = time.time()
    df = extraer(cfg, n, 20240101, HASTA)
    print(f"{n:22s} {len(df):>10,d} filas  {time.time()-t:6.1f}s  cols={list(df.columns)[:6]}")
