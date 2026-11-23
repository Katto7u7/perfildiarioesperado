"""Persistencia transaccional e idempotente.

- id_politica_run = PRS-{fecha_corte}-{version}-{hash8(config + huella de insumos)}: determinista.
- Si el id ya existe: no se reescribe (idempotente) salvo force=True (DELETE + INSERT de ESE id en la misma transacción).
- Destino real (decisión del usuario 2026-11-20): config destino.sqlite (DATABASES_LEGACY/bancoDBPlanificacion_minimo.db).
  Las 2 tablas de perfil ya existen; las 3 complementarias se crean solo si no existen. El catálogo SUCURSAL no se toca.
- Pruebas: base SQLite indicada con --db (por defecto out/dev.db) con el DDL destino idéntico.
- Postgres: perfil_retiro/postgres_sync.py (config destino.postgres.habilitado).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from .config import RAIZ
from .contrato import COLS_CORTE, COLS_PERFIL

DDL_COMPLEMENTO = """
CREATE TABLE IF NOT EXISTS PERFIL_RUN (
    id_politica_run TEXT PRIMARY KEY, fecha_corte INTEGER NOT NULL, fecha_objetivo INTEGER NOT NULL,
    regimen TEXT NOT NULL, version_modelo TEXT NOT NULL, config_hash TEXT NOT NULL, huella_insumos TEXT NOT NULL,
    parametros TEXT NOT NULL, estado TEXT NOT NULL CHECK (estado IN ('OK','OK_CON_RECHAZOS','FALLIDA')),
    n_perfiles INTEGER NOT NULL, n_cortes INTEGER NOT NULL, n_exclusiones INTEGER NOT NULL, n_predicciones INTEGER NOT NULL,
    duracion_seg REAL, mensaje TEXT, vigente INTEGER NOT NULL DEFAULT 1,
    modo TEXT NOT NULL DEFAULT 'SOMBRA' CHECK (modo IN ('SOMBRA','PRODUCCION','HISTORICO','PRUEBA')),
    pg_sincronizado INTEGER NOT NULL DEFAULT 0 CHECK (pg_sincronizado IN (0,1)),
    created_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS PERFIL_EXCLUSION (
    id_politica_run TEXT NOT NULL, id_sucursal INTEGER NOT NULL, motivo TEXT NOT NULL, detalle TEXT,
    PRIMARY KEY (id_politica_run, id_sucursal, motivo));
CREATE TABLE IF NOT EXISTS PREDICCION_RETIRO_DIARIA (
    id_politica_run TEXT NOT NULL, id_sucursal INTEGER NOT NULL, fecha_corte INTEGER NOT NULL, fecha_objetivo INTEGER NOT NULL,
    horizonte_dias INTEGER NOT NULL, retiro_esperado REAL NOT NULL, retiro_p10 REAL NOT NULL, retiro_p50 REAL NOT NULL,
    retiro_p90 REAL NOT NULL, retiro_p95 REAL NOT NULL, prob_retiro_extraordinario REAL, saldo_inicial REAL,
    umbral_critico REAL NOT NULL, prob_saldo_critico REAL, prob_quiebre_intradia REAL, corte_presion INTEGER, prob_quiebre_900 REAL,
    prob_quiebre_1100 REAL, prob_quiebre_1300 REAL, prob_quiebre_1700 REAL, frac_900 REAL, frac_1100 REAL, frac_1300 REAL,
    dias_cobertura_p50 REAL, dias_cobertura_p90 REAL, nivel_confianza TEXT NOT NULL, factores TEXT,
    created_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (id_politica_run, id_sucursal, fecha_objetivo),
    CHECK (retiro_p10 <= retiro_p50 AND retiro_p50 <= retiro_p90 AND retiro_p90 <= retiro_p95));
"""


def id_run(fecha_corte: int, version: str, config_hash: str, huella: dict) -> str:
    h = hashlib.sha256((config_hash + json.dumps(huella, sort_keys=True)).encode()).hexdigest()[:8]
    return f"PRS-{fecha_corte}-{version}-{h}"


def ddl_destino(cfg) -> str:
    """DDL de PERFIL_RETIRO_SUCURSAL y _CORTE tomado TEXTUALMENTE del dump de referencia."""
    src = Path(RAIZ / ".." / "Velocidades_20260616" / "bancoDBPlanificacion_minimo.sql").resolve().read_text(encoding="utf-8")
    bloques = re.findall(r"CREATE TABLE (?:IF NOT EXISTS )?\"?(PERFIL_RETIRO_SUCURSAL(?:_CORTE)?|SUCURSAL)\"? \(.*?\n\);", src, re.S)
    stm = re.findall(r"(CREATE TABLE (?:IF NOT EXISTS )?\"?(?:PERFIL_RETIRO_SUCURSAL(?:_CORTE)?|SUCURSAL)\"? \(.*?\n\);)", src, re.S)
    stm = [re.sub(r"CREATE TABLE (IF NOT EXISTS )?", "CREATE TABLE IF NOT EXISTS ", x) for x in stm]
    assert len(stm) == 3, f"DDL destino no encontrado completo: {bloques}"
    return "\n".join(stm)


COLS_DESTINO_REQUERIDAS = {"PERFIL_RETIRO_SUCURSAL": COLS_PERFIL, "PERFIL_RETIRO_SUCURSAL_CORTE": COLS_CORTE}


def ruta_destino(cfg) -> Path:
    return (RAIZ / cfg["destino"]["sqlite"]).resolve()


def respaldos_existentes(path: Path) -> list[Path]:
    """Respaldos propios de este proceso (<base>.bak_perfil_AAAAMMDD), del más viejo al más nuevo."""
    patron = re.compile(re.escape(path.name) + r"\.bak_perfil_\d{8}$")
    return sorted(p for p in path.parent.iterdir() if patron.fullmatch(p.name))


def respaldar(path: Path, conservar: int = 5, hoy: dt.date | None = None) -> Path | None:
    """Copia del archivo destino antes de la PRIMERA escritura del día (una por fecha).

    - Verifica espacio libre (>= 1.5 x tamaño de la base + 1 GB); si no alcanza, NO escribe (excepción).
    - Retención: conserva los `conservar` respaldos más recientes; solo borra archivos <base>.bak_perfil_AAAAMMDD
      creados por este proceso, nunca otros respaldos de la carpeta.
    """
    import shutil
    path = Path(path)
    bak = path.with_name(path.name + f".bak_perfil_{(hoy or dt.date.today()):%Y%m%d}")
    if bak.exists():
        return None
    libre = shutil.disk_usage(path.parent).free
    necesario = int(1.5 * path.stat().st_size) + 2**30
    if libre < necesario:
        raise RuntimeError(f"Espacio libre insuficiente para respaldar {path.name}: {libre / 2**30:.1f} GB libres, "
                           f"se requieren {necesario / 2**30:.1f} GB. No se escribe sin respaldo.")
    src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(bak)
    with dst:
        src.backup(dst)                         # respaldo consistente aunque haya otros lectores
    src.close(); dst.close()
    for viejo in respaldos_existentes(path)[:-max(1, int(conservar))]:
        viejo.unlink()
    return bak


def conectar_destino(cfg, path: Path | None = None) -> sqlite3.Connection:
    """Base real: verifica tablas destino (no las crea), crea complementarias si faltan, FK activas."""
    path = Path(path) if path else ruta_destino(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Destino no encontrado: {path}")
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA busy_timeout = 60000")
    con.execute("PRAGMA foreign_keys = ON")
    for t, cols in COLS_DESTINO_REQUERIDAS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
        if not have:
            con.close()
            raise RuntimeError(f"La tabla destino {t} no existe en {path.name}: no se crea automáticamente")
        falt = set(cols) - have
        if falt:
            con.close()
            raise RuntimeError(f"La tabla {t} no tiene las columnas esperadas: {sorted(falt)}")
    con.executescript(DDL_COMPLEMENTO)
    return con


def sucursales_catalogo(con) -> set:
    return {r[0] for r in con.execute('SELECT id_sucursal FROM "SUCURSAL"')}


def conectar_dev(cfg, path: Path | None = None) -> sqlite3.Connection:
    path = path or RAIZ / "out" / "dev.db"
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ddl_destino(cfg))
    con.executescript(DDL_COMPLEMENTO)
    return con


def escribir(con, run: dict, perfil: pd.DataFrame, corte: pd.DataFrame, excl: pd.DataFrame,
             pred: pd.DataFrame | None, sucursal: pd.DataFrame | None = None, force: bool = False) -> str:
    """Escritura atómica. Devuelve 'ESCRITO', 'OMITIDO_YA_EXISTE' o lanza excepción (rollback)."""
    rid = run["id_politica_run"]
    cur = con.cursor()
    existe = cur.execute("SELECT COUNT(*) FROM PERFIL_RUN WHERE id_politica_run=?", (rid,)).fetchone()[0]
    if existe and not force:
        return "OMITIDO_YA_EXISTE"
    try:
        cur.execute("BEGIN")
        if sucursal is not None:                # SOLO en la base de pruebas: completa el catálogo para las FK
            cur.executemany('INSERT OR IGNORE INTO "SUCURSAL"(id_sucursal, sucursal) VALUES (?, ?)',
                            sucursal[["id_sucursal", "sucursal"]].itertuples(index=False, name=None))
        if existe:
            for t in ("PERFIL_RETIRO_SUCURSAL_CORTE", "PERFIL_RETIRO_SUCURSAL", "PERFIL_EXCLUSION",
                      "PREDICCION_RETIRO_DIARIA", "PERFIL_RUN"):
                cur.execute(f"DELETE FROM {t} WHERE id_politica_run=?", (rid,))
        cur.execute("UPDATE PERFIL_RUN SET vigente=0 WHERE fecha_corte=?", (run["fecha_corte"],))
        ins = lambda t, cols, df: cur.executemany(
            f"INSERT INTO {t} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            [tuple(None if pd.isna(x) else (x.item() if hasattr(x, 'item') else x) for x in r)
             for r in df[cols].itertuples(index=False, name=None)])
        ins("PERFIL_RETIRO_SUCURSAL", COLS_PERFIL, perfil)
        ins("PERFIL_RETIRO_SUCURSAL_CORTE", COLS_CORTE, corte)
        ins("PERFIL_EXCLUSION", ["id_politica_run", "id_sucursal", "motivo", "detalle"], excl.drop_duplicates(["id_sucursal", "motivo"]))
        if pred is not None and len(pred):
            pcols = [c for c in pred.columns if c != "created_at_utc"]
            ins("PREDICCION_RETIRO_DIARIA", pcols, pred)
        rcols = list(run.keys())
        cur.execute(f"INSERT INTO PERFIL_RUN ({','.join(rcols)}) VALUES ({','.join('?'*len(rcols))})", tuple(run.values()))
        con.commit()
        return "ESCRITO"
    except Exception:
        con.rollback()
        raise


def ahora_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
