"""Extracción de fuentes. TODO acceso es de solo lectura y acotado por fecha de corte.

Regla anti-fuga: toda tabla de hechos se lee con `id_fecha <= fecha_corte`.
Solo FECHA y DISPERSION (calendarios publicados por adelantado) se leen completos.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

from .config import RAIZ, Config

log = logging.getLogger(__name__)
SQL_DIR = RAIZ / "sql"
CACHE_DIR = RAIZ / "out" / "cache"

# consulta -> (clave de base, ¿acotada por fecha?)
CONSULTAS = {
    "movimientos_diario": ("db11", True),
    "monitoreo_cortes": ("db_saldos", True),
    "saldos": ("db_saldos", True),
    "retiros": ("db_saldos", True),
    "fecha": ("db11", False),
    "dispersion": ("db11", False),
    "sucursal": ("db11", False),
    "gestionada": ("db_operativo", True),
    "prediccion_ds": ("db_operativo", True),
    "politica_visita": ("db_visita", False),
    "ingvar_monto": ("db_ingvar", True),
}


def conectar_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    uri = "file:" + str(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def leer_sql(nombre: str) -> str:
    return (SQL_DIR / f"{nombre}.sql").read_text(encoding="utf-8")


def extraer(cfg: Config, nombre: str, desde: int = 19000101, hasta: int = 29991231,
            usar_cache: bool = True) -> pd.DataFrame:
    """Ejecuta una consulta parametrizada. `hasta` es la fecha de corte (inclusive)."""
    clave, acotada = CONSULTAS[nombre]
    sql = leer_sql(nombre)
    tag = f"{nombre}_{desde}_{hasta}" if acotada else nombre
    cache = CACHE_DIR / f"{tag}.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)
    t0 = time.time()
    con = conectar_ro(cfg.ruta(clave))
    try:
        params = {"desde": desde, "hasta": hasta} if acotada else {}
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()
    log.info("extraer %s: %d filas en %.1fs", nombre, len(df), time.time() - t0)
    if usar_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def leer_tope(cfg: Config) -> pd.DataFrame:
    t = pd.read_csv(cfg.ruta("tope_csv"), encoding="utf-8-sig")
    t.columns = [c.strip() for c in t.columns]
    return t.rename(columns={"CC": "id_sucursal", "Tope": "capacidad_boveda"})[["id_sucursal", "capacidad_boveda"]]


def leer_politica(cfg: Config, juego: str = "vigente") -> pd.DataFrame | None:
    p = cfg.ruta("politica_dir") / f"politica_{juego}.csv"
    return pd.read_csv(p) if p.exists() else None


def huella_insumos(dfs: dict[str, pd.DataFrame], fecha_col: str = "id_fecha") -> dict:
    """Huella reproducible: filas, fecha máxima y hash del contenido por insumo."""
    out = {}
    for k, df in dfs.items():
        h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]
        mx = int(df[fecha_col].max()) if fecha_col in df.columns and len(df) else None
        out[k] = {"filas": int(len(df)), "max_fecha": mx, "hash": h}
    return out
