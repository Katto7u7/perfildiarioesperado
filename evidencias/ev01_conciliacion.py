"""EV-01 Conciliación total de fuentes de retiro diario y detección de anomalías de carga.

Pregunta: ¿MOVIMIENTOS (fuente de verdad de la demanda) es completo y consistente día a día?
¿Qué días están dañados (faltantes, copias, parciales) y cómo se relaciona con las demás fuentes?
"""
import numpy as np
import pandas as pd

from comun import calendario, fuente, guardar_json, salida

EV = "EV-01"
out = salida(EV)
cal = calendario()

# ---------------------------------------------------------------- MOVIMIENTOS
mv = fuente("movimientos_diario")
dup = mv[mv.n_filas > 1]
dia = mv.groupby(["id_fecha", "id_sucursal"], as_index=False).agg(mov=("monto", "sum"), n_prog=("id_programa", "nunique"))
porfecha = dia.groupby("id_fecha").agg(n_suc=("id_sucursal", "nunique"), total=("mov", "sum"), n_pos=("mov", lambda s: (s > 0).sum())).reset_index()
porfecha = porfecha.merge(cal[["id_fecha", "fecha", "dow", "is_holiday", "habil_cal"]], on="id_fecha", how="left")

# Copias: vector (sucursal -> monto) idéntico al día cargado anterior
piv = dia.pivot(index="id_sucursal", columns="id_fecha", values="mov")
fechas = list(piv.columns)
copias = []
for a, b in zip(fechas[:-1], fechas[1:]):
    x, y = piv[a], piv[b]
    m = x.notna() & y.notna() & ((x > 0) | (y > 0))
    if m.sum() < 100:
        continue
    eq = (np.abs(x[m] - y[m]) < 0.01).mean()
    if eq > 0.5:
        copias.append({"fecha": int(b), "copia_de": int(a), "pct_identico": round(float(eq), 4), "n_suc": int(m.sum())})
copias = pd.DataFrame(copias)

# Días hábiles de calendario sin carga (desde inicio útil)
ini = 20250602
hab = cal[(cal.habil_cal) & (cal.id_fecha >= ini) & (cal.id_fecha <= int(porfecha.id_fecha.max()))]
cargados = set(porfecha.loc[porfecha.n_pos > 500, "id_fecha"])
faltantes = hab[~hab.id_fecha.isin(cargados)][["id_fecha", "fecha", "dow"]]
# Fechas cargadas que NO son hábiles de calendario (fines de semana / feriados con movimiento)
no_hab = porfecha[(porfecha.id_fecha >= ini) & (~porfecha.habil_cal.fillna(False)) & (porfecha.n_pos > 0)]

# ---------------------------------------------------------------- otras fuentes
rt = fuente("retiros").rename(columns={"monto_total": "ret"})[["id_fecha", "id_sucursal", "ret"]]
iv = fuente("ingvar_monto").rename(columns={"monto": "iv"})
sa = fuente("saldos").rename(columns={"retiros": "sal"})[["id_fecha", "id_sucursal", "sal"]]
mon = fuente("monitoreo_cortes")
m17 = mon[mon.hora_corte == 1700][["id_fecha", "id_sucursal", "retiros"]].rename(columns={"retiros": "mon17"})

base = dia[["id_fecha", "id_sucursal", "mov"]]
comp = base.merge(rt, on=["id_fecha", "id_sucursal"], how="outer").merge(iv, on=["id_fecha", "id_sucursal"], how="outer") \
    .merge(m17, on=["id_fecha", "id_sucursal"], how="outer").merge(sa, on=["id_fecha", "id_sucursal"], how="outer")
comp = comp.merge(cal[["id_fecha", "habil_cal"]], on="id_fecha", how="left")
comp = comp[(comp.habil_cal == True) & (comp.id_fecha >= 20240101)]  # noqa: E712
comp["mes"] = comp.id_fecha // 100


def resumen_par(a, b, nombre):
    c = comp[[a, b, "mes", "id_fecha"]].copy()
    # en la era dispersa, sucursal sin fila en MOVIMIENTOS en día cargado = 0
    if a == "mov":
        c.loc[c[a].isna() & c.id_fecha.isin(cargados), a] = 0.0
    c = c.dropna(subset=[a, b])
    c["exacto"] = (c[a] - c[b]).abs() < 1
    c["reldif"] = (c[b] - c[a]) / c[a].where(c[a] > 0)
    g = c.groupby("mes").agg(n=("exacto", "size"), pct_exacto=("exacto", "mean"), suma_a=(a, "sum"), suma_b=(b, "sum"),
                             reldif_mediana=("reldif", "median"))
    g["razon_suma_b_a"] = g.suma_b / g.suma_a
    g.insert(0, "comparacion", nombre)
    return g.reset_index()


tabla = pd.concat([
    resumen_par("mov", "ret", "MOVIMIENTOS vs RETIROS"),
    resumen_par("mov", "iv", "MOVIMIENTOS vs INGENIERIA_VARIABLES"),
    resumen_par("mov", "mon17", "MOVIMIENTOS vs MONITOREO@1700"),
    resumen_par("mov", "sal", "MOVIMIENTOS vs SALDOS.retiros"),
    resumen_par("mon17", "sal", "MONITOREO@1700 vs SALDOS.retiros"),
])
tabla.to_csv(out / "conciliacion_mensual.csv", index=False)

# Clasificación de discrepancias MOV vs MON@1700 (días hábiles, desde inicio útil)
c = comp[(comp.id_fecha >= ini)].copy()
c.loc[c.mov.isna() & c.id_fecha.isin(cargados), "mov"] = 0.0
c = c.dropna(subset=["mon17"])
fech_copia = set(copias.fecha) if len(copias) else set()
tol, tol_abs = 0.05, 50_000
d = c.mon17 - c.mov
c["clase"] = np.select(
    [~c.id_fecha.isin(cargados), c.id_fecha.isin(fech_copia), c.mov.isna(),
     d.abs() < 1, (d.abs() <= np.maximum(tol * c.mov, tol_abs)),
     d > np.maximum(tol * c.mov, tol_abs), d < -np.maximum(tol * c.mov, tol_abs)],
    ["CARGA_FALTANTE", "COPIA", "SIN_FILA", "EXACTO", "DENTRO_TOLERANCIA", "MON_MAYOR(traspaso/otros)", "MOV_MAYOR"],
    default="OTRO")
clases = c.groupby("clase").agg(n=("clase", "size"), monto_dif=("mon17", lambda s: 0)).drop(columns="monto_dif")
clases["pct"] = clases.n / clases.n.sum()
clases_mes = pd.crosstab(c.mes, c.clase, normalize="index").round(4)
clases_mes.to_csv(out / "clases_discrepancia_mensual.csv")
# MOV_MAYOR: retiro de clientes mayor que salida total -> imposible contablemente -> investigar
mm = c[c.clase == "MOV_MAYOR"]
mm_top = mm.assign(dif=mm.mov - mm.mon17).sort_values("dif", ascending=False).head(30)
mm_top.to_csv(out / "top_mov_mayor_que_monitoreo.csv", index=False)

# Parciales vs traspasos masivos.
# Una carga PARCIAL de MOVIMIENTOS baja la razón MOV/MON de la sucursal *típica* (mediana por sucursal).
# Un TRASPASO masivo baja la razón de los totales, pero la sucursal mediana sigue en ~1.
c["r_suc"] = c.mov / c.mon17.where(c.mon17 > 0)
rz = c.groupby("id_fecha").agg(mov=("mov", "sum"), mon=("mon17", "sum"), r_mediana_suc=("r_suc", "median"),
                                r_p25_suc=("r_suc", lambda s: s.quantile(0.25)), pct_exacto=("clase", lambda s: (s == "EXACTO").mean()))
rz["razon_totales"] = rz.mov / rz.mon
med, mad = rz.razon_totales.median(), (rz.razon_totales - rz.razon_totales.median()).abs().median()
anom = rz[(rz.razon_totales < med - 6 * 1.4826 * mad) | (rz.r_p25_suc < 0.9)].copy()
anom["diagnostico"] = np.select(
    [anom.mov == 0, anom.r_mediana_suc < 0.9, anom.r_p25_suc < 0.97],
    ["CARGA_FALTANTE", "PARCIAL", "PARCIAL_LEVE_SOSPECHOSO"], default="TRASPASO_MASIVO_EN_MONITOREO")
parciales = anom.reset_index()
rz.reset_index().to_csv(out / "razon_mov_mon_por_fecha.csv", index=False)

res = {
    "movimientos": {
        "filas_agregadas": len(mv), "rango": [int(mv.id_fecha.min()), int(mv.id_fecha.max())],
        "llaves_duplicadas(suc,prog,dia)": int(len(dup)),
        "llaves_duplicadas_con_montos_distintos": int((dup.n_montos_distintos > 1).sum()),
        "monto_negativo": int((mv.monto < 0).sum()),
        "max_fila": float(mv.monto.max()),
    },
    "copias_detectadas": copias.to_dict("records"),
    "dias_habiles_sin_carga_desde_inicio_util": [int(x) for x in faltantes.id_fecha],
    "fechas_no_habiles_con_movimiento": no_hab[["id_fecha", "n_pos", "total"]].to_dict("records"),
    "parciales_por_razon_mov_mon": parciales.to_dict("records"),
    "razon_mov_mon_mediana": float(med),
    "clases_discrepancia_mov_vs_mon17": clases.reset_index().to_dict("records"),
}
guardar_json(EV, res)
porfecha.to_csv(out / "carga_por_fecha.csv", index=False)
dup.to_csv(out / "llaves_duplicadas.csv", index=False)

pd.set_option("display.width", 200)
print(pd.Series(res["movimientos"]))
print("\nCOPIAS:\n", copias)
print("\nHÁBILES SIN CARGA:", res["dias_habiles_sin_carga_desde_inicio_util"])
print("\nNO HÁBILES CON MOV:\n", no_hab[["id_fecha", "dow", "is_holiday", "n_pos", "total"]].to_string())
print("\nPARCIALES:\n", parciales)
print("\nCLASES:\n", clases)
print("\nCONCILIACIÓN (meses seleccionados):")
print(tabla[tabla.mes.isin([202406, 202501, 202506, 202509, 202512, 202603, 202606, 202609, 202610, 202611])].round(4).to_string())
