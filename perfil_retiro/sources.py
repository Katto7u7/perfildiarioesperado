"""Extracción de fuentes. TODO acceso es de solo lectura y acotado por fecha de corte.

El origen lo define FUENTE en el .env del proyecto (o la variable de entorno): `sqlite` (por defecto, archivos .db de
config.yaml, consultas en sql/) o `postgres` (consultas en sql/fuentes_pg/).

Regla anti-fuga: toda tabla de hechos se lee con `id_fecha <= fecha_corte`.
Solo FECHA y DISPERSION (calendarios publicados por adelantado) se leen completos.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

from .config import RAIZ, Config
from .postgres_sync import connection_kwargs, leer_dotenv

log = logging.getLogger(__name__)
SQL_DIR = RAIZ / "sql"
SQL_PG_DIR = SQL_DIR / "fuentes_pg"
RUTA_ENV = RAIZ / ".env"
FUENTES = ("sqlite", "postgres")
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
    "programas_emergentes": ("db_operativo", False),
}


def entorno() -> dict:
    """Variables de lectura: el .env del proyecto sobre el entorno del proceso (FUENTE: gana la del proceso).

    No escribe en os.environ: las credenciales de lectura no se mezclan con las del destino (destino.postgres.env).
    """
    env = {**os.environ, **(leer_dotenv(RUTA_ENV) if RUTA_ENV.exists() else {})}
    if os.environ.get("FUENTE"):
        env["FUENTE"] = os.environ["FUENTE"]
    return env


def fuente_activa(env: dict | None = None) -> str:
    f = ((entorno() if env is None else env).get("FUENTE") or "sqlite").strip().lower()
    if f not in FUENTES:
        raise ValueError(f"FUENTE={f!r} no válida: usa {' o '.join(FUENTES)}")
    return f


def conectar_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    uri = "file:" + str(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def leer_sql(nombre: str, carpeta: Path = SQL_DIR) -> str:
    return (carpeta / f"{nombre}.sql").read_text(encoding="utf-8")


def _leer_sqlite(path: Path, nombre: str, params: dict) -> pd.DataFrame:
    con = conectar_ro(path)
    try:
        return pd.read_sql_query(leer_sql(nombre), con, params=params)
    finally:
        con.close()


def _leer_postgres(env: dict, nombre: str, params: dict) -> pd.DataFrame:
    import psycopg2

    ka = connection_kwargs(env)
    dsn = ka.pop("dsn", None)
    con = psycopg2.connect(dsn, connect_timeout=15, **ka) if dsn else psycopg2.connect(connect_timeout=15, **ka)
    try:
        con.set_session(readonly=True)
        with con.cursor() as cur:
            cur.execute(leer_sql(nombre, SQL_PG_DIR), params or None)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame.from_records(cur.fetchall(), columns=cols, coerce_float=True)
    finally:
        con.close()


def agrupar_dispersion(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (fecha, programa) con sus letras juntas; las filas repetidas cuentan una sola vez.

    `letras` / `letras_segundo`: letras (separadas por coma) en su primer / segundo día; letra nula = "?".
    `n_letras` no cuenta las nulas.
    """
    llave = ["id_fecha", "id_programa"]
    x = d.assign(_l=d.letra.fillna("?").astype(str), _2=d.is_second_letterday == 1,
                 _1=(d.is_second_letterday == 0).astype(int))
    out = x.groupby(llave).agg(
        n_letras=("letra", "nunique"), primero=("_1", "max"), segundo=("is_second_letterday", "max"))
    for col, m in (("letras", ~x["_2"]), ("letras_segundo", x["_2"])):
        juntas = x[m].groupby(llave)["_l"].agg(lambda s: ",".join(sorted(set(s))))
        out[col] = juntas.reindex(out.index).fillna("")
    return out.reset_index()[llave + ["letras", "letras_segundo", "n_letras", "primero", "segundo"]]


def extraer(cfg: Config, nombre: str, desde: int = 19000101, hasta: int = 29991231,
            usar_cache: bool = True) -> pd.DataFrame:
    """Ejecuta una consulta parametrizada. `hasta` es la fecha de corte (inclusive)."""
    clave, acotada = CONSULTAS[nombre]
    env = entorno()
    origen = fuente_activa(env)
    if origen == "postgres" and not (SQL_PG_DIR / f"{nombre}.sql").exists():
        raise NotImplementedError(f"La consulta '{nombre}' no tiene versión Postgres (solo la usan evidencias/): "
                                  "ejecútala con FUENTE=sqlite")
    tag = f"{nombre}_{desde}_{hasta}" if acotada else nombre
    cache = CACHE_DIR / f"{'pg_' if origen == 'postgres' else ''}{tag}.parquet"
    if usar_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        t0 = time.time()
        params = {"desde": desde, "hasta": hasta} if acotada else {}
        if origen == "postgres":
            df = _leer_postgres(env, nombre, params)
        else:
            df = _leer_sqlite(cfg.ruta(clave), nombre, params)
        log.info("extraer %s (%s): %d filas en %.1fs", nombre, origen, len(df), time.time() - t0)
        if usar_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)          # se guarda tal como viene de la fuente
    return agrupar_dispersion(df) if nombre == "dispersion" else df


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
