"""Volcado SQLite (bancoDBPlanificacion_minimo.db) -> Postgres (base `sucursales`, esquema `planeacion`).

Patrón tomado de kilometraje/actualizar_postgres_politica_km.py: .env con PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD
(o DATABASE_URL), una sola transacción, DRY_RUN = ROLLBACK.

Reglas:
- Las tablas de perfil (perfil_retiro_sucursal, perfil_retiro_sucursal_corte) DEBEN existir en Postgres: no se crean.
- Las 3 complementarias deben existir (las crea el DBA: sql/postgres/01_...); solo si faltan se intenta crearlas.
- Antes de insertar se verifica que todas las sucursales existan en planeacion.sucursal (FK).
- La vigencia se propaga: corridas reemplazadas en SQLite (vigente = 0) quedan vigente = 0 en Postgres.
- Se vuelcan solo las corridas con pg_sincronizado = 0 que todavía no existan en Postgres (idempotente).
- Conteos SQLite = Postgres por corrida o ROLLBACK total. Solo tras COMMIT se marca pg_sincronizado = 1 en SQLite.
- COMMIT solo si config destino.postgres.habilitado = true; --dry-run siempre termina en ROLLBACK.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd

from .config import RAIZ
from .persistencia import conectar_destino, ruta_destino

log = logging.getLogger("perfil_retiro")

TABLAS_PERFIL = ["PERFIL_RETIRO_SUCURSAL", "PERFIL_RETIRO_SUCURSAL_CORTE"]
TABLAS_COMPLEMENTO = ["PERFIL_RUN", "PERFIL_EXCLUSION", "PREDICCION_RETIRO_DIARIA"]
PK = {"PERFIL_RETIRO_SUCURSAL": ["id_politica_run", "id_sucursal"],
      "PERFIL_RETIRO_SUCURSAL_CORTE": ["id_politica_run", "id_sucursal", "hora_inicio", "hora_fin"],
      "PERFIL_RUN": ["id_politica_run"], "PERFIL_EXCLUSION": ["id_politica_run", "id_sucursal", "motivo"],
      "PREDICCION_RETIRO_DIARIA": ["id_politica_run", "id_sucursal", "fecha_objetivo"]}
TIPO_PG = {"INTEGER": "integer", "REAL": "double precision", "TEXT": "text", "NUMERIC": "double precision"}


def leer_dotenv(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de credenciales {path}")
    valores = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        valores.setdefault(k.strip(), v)
    return valores


def load_dotenv(path: Path) -> None:
    for k, v in leer_dotenv(path).items():
        os.environ.setdefault(k, v)


def connection_kwargs(env=None) -> dict:
    env = os.environ if env is None else env
    if env.get("DATABASE_URL"):
        return {"dsn": env.get("DATABASE_URL")}
    falt = [n for n in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD") if not env.get(n)]
    if falt:
        raise SystemExit("Faltan variables de conexión: " + ", ".join(falt))
    return {"host": env.get("PGHOST"), "port": int(env.get("PGPORT") or 5432), "dbname": env.get("PGDATABASE"),
            "user": env.get("PGUSER"), "password": env.get("PGPASSWORD"), "sslmode": env.get("PGSSLMODE") or "prefer"}


def ddl_pg(con_sqlite, tabla: str, esquema: str) -> str:
    """CREATE TABLE IF NOT EXISTS en Postgres a partir del esquema SQLite (tipos traducidos, misma PK)."""
    cols = []
    for _, nom, tipo, notnull, _, _ in con_sqlite.execute(f"PRAGMA table_info({tabla})"):
        t = "timestamp without time zone" if nom == "created_at_utc" else TIPO_PG.get(tipo.upper(), "text")
        cols.append(f'"{nom}" {t}{" NOT NULL" if notnull else ""}')
    pk = ", ".join(f'"{c}"' for c in PK[tabla])
    return f'CREATE TABLE IF NOT EXISTS "{esquema}"."{tabla.lower()}" ({", ".join(cols)}, PRIMARY KEY ({pk}))'


def _a_timestamp(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return dt.datetime.fromisoformat(str(v).replace("Z", ""))


def sincronizar(cfg, commit: bool = False, sqlite_path: Path | None = None) -> dict:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values

    pgc = cfg["destino"]["postgres"]
    if commit and not pgc.get("habilitado"):
        raise PermissionError("Volcado a Postgres apagado (destino.postgres.habilitado = false). Usa --dry-run.")
    load_dotenv((RAIZ / pgc["env"]).resolve())
    esquema = pgc.get("esquema") or os.getenv("PGSCHEMA", "planeacion")
    src = conectar_destino(cfg, sqlite_path or ruta_destino(cfg))
    runs = pd.read_sql("SELECT id_politica_run FROM PERFIL_RUN WHERE pg_sincronizado = 0 ORDER BY fecha_corte", src).id_politica_run.tolist()
    res = {"esquema": esquema, "corridas_pendientes": len(runs), "dry_run": not commit, "por_tabla": {}}
    ka = connection_kwargs()
    pg = psycopg2.connect(ka["dsn"]) if "dsn" in ka else psycopg2.connect(**ka)
    pg.autocommit = False
    try:
        with pg.cursor() as cur:
            for t in TABLAS_PERFIL:
                cur.execute("SELECT to_regclass(%s)", (f"{esquema}.{t.lower()}",))
                if cur.fetchone()[0] is None:
                    raise RuntimeError(f"No existe {esquema}.{t.lower()} en Postgres: no se crea automáticamente")
            for t in TABLAS_COMPLEMENTO:
                # Solo se intenta crear si falta: Postgres exige permiso CREATE en el esquema aun con IF NOT EXISTS,
                # y el usuario de la aplicación normalmente no lo tiene (las crea el DBA con sql/postgres/01_...).
                cur.execute("SELECT to_regclass(%s)", (f"{esquema}.{t.lower()}",))
                if cur.fetchone()[0] is None:
                    cur.execute(ddl_pg(src, t, esquema))
            ya = set()
            if runs:
                cur.execute(sql.SQL("SELECT id_politica_run FROM {} WHERE id_politica_run = ANY(%s)").format(
                    sql.Identifier(esquema, "perfil_run")), (runs,))
                ya = {r[0] for r in cur.fetchall()}
            nuevas = [r for r in runs if r not in ya]
            res["corridas_ya_en_postgres"] = sorted(ya)
            res["corridas_a_volcar"] = nuevas
            if nuevas:
                # FK de perfil_retiro_sucursal -> planeacion.sucursal: se verifica ANTES para dar un error claro
                suc = pd.read_sql(f"SELECT DISTINCT id_sucursal FROM PERFIL_RETIRO_SUCURSAL WHERE id_politica_run IN "
                                  f"({','.join('?' * len(nuevas))})", src, params=nuevas).id_sucursal.astype(int).tolist()
                cur.execute(sql.SQL("SELECT id_sucursal FROM {} WHERE id_sucursal = ANY(%s)").format(
                    sql.Identifier(esquema, "sucursal")), (suc,))
                faltan = sorted(set(suc) - {int(r[0]) for r in cur.fetchall()})
                if faltan:
                    raise RuntimeError(f"{len(faltan)} sucursales no existen en {esquema}.sucursal (FK): {faltan[:20]}... "
                                       "Darlas de alta en el catálogo de Postgres o excluirlas; no se vuelca nada.")
            for t in TABLAS_PERFIL + TABLAS_COMPLEMENTO:
                if not nuevas:
                    break
                cols_sqlite = [r[1] for r in src.execute(f"PRAGMA table_info({t})")]
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",
                            (esquema, t.lower()))
                cols_pg = {r[0] for r in cur.fetchall()}
                falt = [c for c in cols_sqlite if c.lower() not in cols_pg]
                if falt:
                    raise RuntimeError(f"{esquema}.{t.lower()} no tiene las columnas {falt}")
                q = f"SELECT {', '.join(cols_sqlite)} FROM {t} WHERE id_politica_run IN ({','.join('?' * len(nuevas))})"
                df = pd.read_sql(q, src, params=nuevas)
                filas = [tuple(_a_timestamp(v) if c == "created_at_utc" else (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                               for c, v in zip(cols_sqlite, r)) for r in df.itertuples(index=False, name=None)]
                if filas:
                    execute_values(cur, sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING").format(
                        sql.Identifier(esquema, t.lower()), sql.SQL(", ").join(map(sql.Identifier, [c.lower() for c in cols_sqlite]))
                    ).as_string(cur), filas, page_size=2000)
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE id_politica_run = ANY(%s)").format(
                    sql.Identifier(esquema, t.lower())), (nuevas,))
                n_pg = cur.fetchone()[0]
                res["por_tabla"][t] = {"sqlite": len(df), "postgres": n_pg}
                if n_pg != len(df):
                    raise RuntimeError(f"Conteo distinto en {t}: SQLite {len(df)} vs Postgres {n_pg}")
            # vigencia: una corrida reemplazada en SQLite (vigente = 0) también deja de estar vigente en Postgres
            no_vig = pd.read_sql("SELECT id_politica_run FROM PERFIL_RUN WHERE vigente = 0", src).id_politica_run.tolist()
            res["vigencia_actualizada"] = 0
            if no_vig:
                cur.execute(sql.SQL("UPDATE {} SET vigente = 0 WHERE vigente = 1 AND id_politica_run = ANY(%s)").format(
                    sql.Identifier(esquema, "perfil_run")), (no_vig,))
                res["vigencia_actualizada"] = cur.rowcount
        if commit:
            pg.commit()
            if nuevas or ya:
                src.executemany("UPDATE PERFIL_RUN SET pg_sincronizado = 1 WHERE id_politica_run = ?", [(r,) for r in nuevas + sorted(ya)])
                src.commit()
            res["resultado"] = "COMMIT"
        else:
            pg.rollback()
            res["resultado"] = "ROLLBACK (dry-run): ningún cambio persistente en Postgres"
    except BaseException:
        pg.rollback()
        raise
    finally:
        pg.close()
        src.close()
    log.info("sync postgres: %s", res)
    return res
