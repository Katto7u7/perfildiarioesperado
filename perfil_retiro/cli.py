"""Línea de comandos.

  python -m perfil_retiro.cli check-sources --fecha-corte 20261113
  python -m perfil_retiro.cli run-daily     --fecha-corte 20261113 [--force] [--db out/dev.db]
  python -m perfil_retiro.cli run-daily     --auto [--excel]  (fecha de corte = último día con cierre en MOVIMIENTOS;
                                                               si ya tiene corrida vigente, no se reprocesa)
  python -m perfil_retiro.cli backfill      --meses 202601-202610   (perfil por cierre de mes, sin predicciones)
  python -m perfil_retiro.cli sync-postgres --dry-run | --commit

Códigos de salida: 0 OK | 2 corrida abortada por calidad/frescura (no se publica) | 1 error inesperado.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import RAIZ, cargar_config


def _log(nivel="INFO"):
    d = RAIZ / "out" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=nivel, format=fmt, handlers=[logging.StreamHandler(sys.stdout),
                                                           logging.FileHandler(d / "perfil_retiro.log", encoding="utf-8")])


def check_sources(cfg, fecha_corte):
    """Diagnóstico de insumos del día de corte, sin escribir nada."""
    from .calidad import detectar_parciales, mov_diario
    from .pipeline import cobertura_monitoreo_1700
    from .sources import extraer
    mv = extraer(cfg, "movimientos_diario", fecha_corte, fecha_corte, usar_cache=False)
    mon = extraer(cfg, "monitoreo_cortes", fecha_corte, fecha_corte, usar_cache=False)
    disp = extraer(cfg, "dispersion", usar_cache=False)
    dia = mov_diario(mv)
    par = detectar_parciales(dia, mon[mon.hora_corte == 1700])
    rep = {"fecha_corte": fecha_corte, "movimientos_filas": len(mv), "sucursales_con_retiro": int((dia.retiro > 0).sum()),
           "monitoreo_cortes": sorted(mon.hora_corte.unique().tolist()),
           "cobertura_monitoreo_1700": round(cobertura_monitoreo_1700(dia, mon[mon.hora_corte == 1700]), 4),
           "minimo_requerido": cfg["calidad"].get("min_cobertura_monitoreo_1700", 0.90),
           "parcial": bool(par.parcial.any()) if len(par) else None,
           "razon_mediana_mov_mon": float(par.r_mediana.iloc[0]) if len(par) else None,
           "dispersion_cargada_hasta": int(disp.id_fecha.max())}
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


def _ultimo_operativo_hasta_hoy(cfg) -> int | None:
    import datetime as dt
    from .panel import calendario_operativo
    from .sources import extraer
    cal = calendario_operativo(extraer(cfg, "fecha", usar_cache=False), extraer(cfg, "dispersion", usar_cache=False))
    hoy = int(dt.date.today().strftime("%Y%m%d"))
    ops = cal[cal.operativo & (cal.id_fecha <= hoy)].id_fecha
    return int(ops.max()) if len(ops) else None


def _corrida_vigente(db, fecha_corte) -> str | None:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
        r = con.execute("SELECT id_politica_run FROM PERFIL_RUN WHERE vigente = 1 AND fecha_corte = ? "
                        "ORDER BY created_at_utc DESC LIMIT 1", (int(fecha_corte),)).fetchone()
        con.close()
        return r[0] if r else None
    except Exception:  # noqa: BLE001  (base sin PERFIL_RUN todavía)
        return None


def backfill(cfg, meses: str, semilla, db, force) -> int:
    """Un perfil por cierre de mes (último día operativo con calidad OK), solo con datos <= ese día."""
    from .panel import calendario_operativo
    from .pipeline import CorridaAbortada, correr_diario
    from .sources import extraer
    lo, hi = (int(x) for x in meses.split("-"))
    cal = calendario_operativo(extraer(cfg, "fecha", usar_cache=False), extraer(cfg, "dispersion", usar_cache=False))
    ops = cal[cal.operativo].id_fecha.sort_values().values
    sem = pd.read_parquet(semilla)
    lg = logging.getLogger("perfil_retiro")
    resultados = []
    for ym in sorted({int(d) // 100 for d in ops if lo <= int(d) // 100 <= hi}):
        candidatos = [int(d) for d in ops if int(d) // 100 == ym][::-1][:3]   # último día del mes; si está dañado, el anterior
        for fc in candidatos:
            try:
                run = correr_diario(cfg, fc, errores_semilla=sem, force=force, db_path=db, solo_perfil=True)
                resultados.append((ym, fc, run["id_politica_run"], run["escritura"], run["n_perfiles"]))
                break
            except CorridaAbortada as e:
                lg.warning("backfill %s: %s; se intenta el día operativo anterior", fc, e)
    for r in resultados:
        print(*r)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="perfil_retiro")
    ap.add_argument("comando", choices=["check-sources", "run-daily", "backfill", "sync-postgres"])
    ap.add_argument("--meses", default="202601-202610")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--excel", action="store_true", help="exporta la corrida a out/entregables/perfil_retiro_<fecha>.xlsx")
    ap.add_argument("--fecha-corte", type=int)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--force", action="store_true", help="reescribe el MISMO id_politica_run (misma huella)")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--semilla-errores", type=Path, default=RAIZ / "out" / "semilla_errores.parquet")
    a = ap.parse_args(argv)
    _log()
    cfg = cargar_config(a.config)
    fc = a.fecha_corte
    if a.comando in ("check-sources", "run-daily") and not a.auto and fc is None:
        print("Indica --fecha-corte AAAAMMDD o --auto")
        return 1
    if a.auto:
        from .sources import extraer
        fc = int(extraer(cfg, "movimientos_diario", 20250101, 29991231, usar_cache=False).id_fecha.max())
        esperado = _ultimo_operativo_hasta_hoy(cfg)
        lg = logging.getLogger("perfil_retiro")
        if esperado and fc < esperado:
            lg.warning("DATOS ATRASADOS: MOVIMIENTOS llega hasta %s pero el último día operativo cerrado es %s; "
                       "se usa %s como fecha de corte", fc, esperado, fc)
        if a.comando == "run-daily" and not a.force:
            from .persistencia import ruta_destino
            ya = _corrida_vigente(a.db or ruta_destino(cfg), fc)
            if ya:
                lg.warning("SIN DATOS NUEVOS: la fecha de corte %s ya tiene corrida vigente (%s); no se reprocesa. "
                           "Para reprocesar a propósito: --fecha-corte %s", fc, ya, fc)
                if a.excel:
                    from .exportar import exportar
                    print("Excel:", exportar(a.db or ruta_destino(cfg), ya))
                return 0
    if a.comando == "check-sources":
        return check_sources(cfg, fc)
    if a.comando == "sync-postgres":
        from .postgres_sync import sincronizar
        if a.commit == a.dry_run:
            print("Indica exactamente uno: --dry-run o --commit")
            return 1
        try:
            print(json.dumps(sincronizar(cfg, commit=a.commit), indent=2, ensure_ascii=False, default=str))
            return 0
        except PermissionError as e:
            print(str(e))
            return 2
    if a.comando == "backfill":
        return backfill(cfg, a.meses, a.semilla_errores, a.db, a.force)
    from .pipeline import CorridaAbortada, correr_diario
    try:
        sem = pd.read_parquet(a.semilla_errores) if a.semilla_errores.exists() else None
        run = correr_diario(cfg, fc, errores_semilla=sem, force=a.force, db_path=a.db)
        print(json.dumps({k: v for k, v in run.items() if k not in ("parametros", "huella_insumos")}, indent=2, ensure_ascii=False))
        if a.excel:
            from .exportar import exportar
            from .persistencia import ruta_destino
            print("Excel:", exportar(a.db or ruta_destino(cfg), run["id_politica_run"]))
        return 0
    except CorridaAbortada as e:
        logging.getLogger("perfil_retiro").error("CORRIDA ABORTADA: %s", e)
        return 2
    except Exception:
        logging.getLogger("perfil_retiro").exception("ERROR INESPERADO")
        return 1


if __name__ == "__main__":
    sys.exit(main())
