"""Destino real y volcado a Postgres (sin conexión a servidor)."""
import sqlite3
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from perfil_retiro.config import cargar_config  # noqa: E402
from perfil_retiro.persistencia import DDL_COMPLEMENTO, conectar_destino, ddl_destino  # noqa: E402
from perfil_retiro.postgres_sync import _a_timestamp, ddl_pg  # noqa: E402

CFG = cargar_config()


def _base(tmp_path, con_tablas=True):
    p = tmp_path / "plan.db"
    con = sqlite3.connect(p)
    if con_tablas:
        con.executescript(ddl_destino(CFG))
    con.close()
    return p


def test_destino_no_crea_tablas_de_perfil(tmp_path):
    p = _base(tmp_path, con_tablas=False)
    with pytest.raises(RuntimeError, match="no existe"):
        conectar_destino(CFG, p)


def test_destino_crea_solo_complementarias(tmp_path):
    p = _base(tmp_path)
    con = conectar_destino(CFG, p)
    tablas = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"PERFIL_RUN", "PERFIL_EXCLUSION", "PREDICCION_RETIRO_DIARIA", "PERFIL_RETIRO_SUCURSAL"} <= tablas
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_ddl_postgres_tipos_y_pk(tmp_path):
    con = sqlite3.connect(":memory:")
    con.executescript(DDL_COMPLEMENTO)
    d = ddl_pg(con, "PREDICCION_RETIRO_DIARIA", "planeacion")
    assert d.startswith('CREATE TABLE IF NOT EXISTS "planeacion"."prediccion_retiro_diaria"')
    assert '"retiro_p50" double precision NOT NULL' in d
    assert '"created_at_utc" timestamp without time zone NOT NULL' in d
    assert 'PRIMARY KEY ("id_politica_run", "id_sucursal", "fecha_objetivo")' in d


def test_timestamp_iso_z():
    t = _a_timestamp("2026-11-18T18:51:36.376Z")
    assert (t.year, t.hour, t.microsecond) == (2026, 18, 376000) and t.tzinfo is None
