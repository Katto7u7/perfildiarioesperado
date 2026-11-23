"""Exporta una corrida a Excel: las mismas tablas de la base, filtradas por id_politica_run.

  python -m perfil_retiro.exportar [--fecha-corte AAAAMMDD | --id-run PRS-...] [--db ruta]
Salida: out/entregables/perfil_retiro_<fecha_corte>.xlsx
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from .config import RAIZ, cargar_config
from .persistencia import ruta_destino

TABLAS = ["PERFIL_RETIRO_SUCURSAL", "PERFIL_RETIRO_SUCURSAL_CORTE", "PREDICCION_RETIRO_DIARIA", "PERFIL_EXCLUSION", "PERFIL_RUN"]


def exportar(db: Path, id_run: str | None = None, fecha_corte: int | None = None, salida: Path | None = None) -> Path:
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    if id_run is None:
        q = "SELECT id_politica_run FROM PERFIL_RUN WHERE vigente = 1"
        q += f" AND fecha_corte = {int(fecha_corte)}" if fecha_corte else ""
        r = con.execute(q + " ORDER BY fecha_corte DESC, created_at_utc DESC LIMIT 1").fetchone()
        if not r:
            raise SystemExit(f"No hay corrida vigente{' para ' + str(fecha_corte) if fecha_corte else ''} en {db}")
        id_run = r[0]
    datos = {t: pd.read_sql(f"SELECT * FROM {t} WHERE id_politica_run = ?", con, params=[id_run]) for t in TABLAS}
    con.close()
    run = datos["PERFIL_RUN"].iloc[0]
    fc = int(run.fecha_corte)
    salida = salida or RAIZ / "out" / "entregables" / f"perfil_retiro_{fc}.xlsx"
    salida.parent.mkdir(parents=True, exist_ok=True)

    pred = datos["PREDICCION_RETIRO_DIARIA"]
    resumen = [
        ("id_politica_run", id_run), ("fecha de corte (datos hasta el cierre de)", fc),
        ("fecha objetivo (día pronosticado)", int(run.fecha_objetivo)), ("régimen", run.regimen),
        ("modo", run.modo + ("  — validación en sombra: no usar como dato oficial sin revisión" if run.modo == "SOMBRA" else "")),
        ("estado", run.estado), ("versión del modelo", run.version_modelo),
        ("perfiles", len(datos["PERFIL_RETIRO_SUCURSAL"])), ("intervalos", len(datos["PERFIL_RETIRO_SUCURSAL_CORTE"])),
        ("predicciones", len(pred)), ("exclusiones", len(datos["PERFIL_EXCLUSION"])),
    ]
    if len(pred):
        resumen += [
            ("retiro esperado total ($)", round(float(pred.retiro_esperado.sum()))),
            ("sucursales con P(quiebre intradía) ≥ 0.5", int((pred.prob_quiebre_intradia >= 0.5).sum())),
            ("sucursales con P(saldo crítico al cierre) ≥ 0.5", int((pred.prob_saldo_critico >= 0.5).sum())),
            ("confianza ALTA / MEDIA / BAJA / INSUFICIENTE",
             " / ".join(str(int((pred.nivel_confianza == k).sum())) for k in ("ALTA", "MEDIA", "BAJA", "INSUFICIENTE"))),
        ]
    tipos = datos["PERFIL_RETIRO_SUCURSAL"].tipo_perfil.value_counts()
    resumen += [(f"tipo_perfil = {k}", int(v)) for k, v in tipos.items()]
    resumen += [("nota", "Las hojas siguientes contienen exactamente las tablas de la base (mismas columnas) para esta corrida."),
                ("diccionario", "prod_perfil_retiro/docs/04_diccionario.md")]

    with pd.ExcelWriter(salida, engine="openpyxl") as xw:
        pd.DataFrame(resumen, columns=["concepto", "valor"]).to_excel(xw, sheet_name="Resumen", index=False)
        for t in TABLAS:
            df = datos[t]
            df.to_excel(xw, sheet_name=t[:31], index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2" if ws.title != "Resumen" else "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, largo + 2), 60)
    return salida


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha-corte", type=int)
    ap.add_argument("--id-run")
    ap.add_argument("--db", type=Path)
    a = ap.parse_args(argv)
    cfg = cargar_config()
    out = exportar(a.db or ruta_destino(cfg), a.id_run, a.fecha_corte)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
