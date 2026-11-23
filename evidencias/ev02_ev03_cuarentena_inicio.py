"""EV-02 Cuarentena y salida no-programa  |  EV-03 Inicio de historia útil.

EV-02: ¿qué sucursal-días se excluyen y por qué? ¿la brecha MON/MOV en quincenas es defecto o comportamiento?
EV-03: ¿desde cuándo MOVIMIENTOS representa la demanda completa? (quiebre estructural)
"""
import numpy as np
import pandas as pd

from comun import bootstrap_ci, fuente, guardar_json, panel_completo, salida
from perfil_retiro.calidad import detectar_parciales, mov_diario

p = panel_completo()
res = {}

# ------------------------------------------------------------------ EV-02
out2 = salida("EV-02")
u = p[p.id_fecha >= 20250602]
est = u.estado.value_counts().rename("n").to_frame()
est["pct"] = est.n / est.n.sum()
est["monto_M"] = u.groupby("estado").retiro.sum() / 1e6
est.to_csv(out2 / "estados_dato.csv")
res["EV-02"] = {"estados": est.reset_index().to_dict("records")}

# Sucursal cerrada: ¿es cierre real? (días consecutivos cerrados por sucursal)
cer = u[u.estado == "SUCURSAL_CERRADA"].groupby("id_sucursal").size().sort_values(ascending=False)
res["EV-02"]["sucursales_con_dias_cerrados"] = int(len(cer))
res["EV-02"]["dias_cerrados_por_sucursal_p50_p90_max"] = [float(cer.median()), float(cer.quantile(.9)), int(cer.max())]

# Salida no-programa: razón MON@1700 / MOV en días válidos SIN traspaso, por contexto
v = u[u.valido & ~u.traspaso & (u.retiro > 0) & (u.n_cortes == 4)].copy()
v["r"] = v.retiros_1700 / v.retiro
v["ctx"] = np.select([v.dias_al_fin_mes == 0, v.es_quincena], ["FIN_DE_MES", "QUINCENA_15"], default="OTRO_DIA")
tab = v.groupby("ctx").agg(n=("r", "size"), r_media=("r", "mean"), r_mediana=("r", "median"),
                           pct_r_mayor_1_01=("r", lambda s: (s > 1.01).mean()),
                           exceso_M=("retiros_1700", "sum"))
tab["exceso_M"] = (v.groupby("ctx").retiros_1700.sum() - v.groupby("ctx").retiro.sum()) / 1e6
tab["exceso_pct_sobre_mov"] = tab.exceso_M * 1e6 / v.groupby("ctx").retiro.sum()
ci = {}
for k, g in v.groupby("ctx"):
    # IC por bootstrap de días (bloque = fecha) de la razón agregada
    d = g.groupby("id_fecha").agg(a=("retiros_1700", "sum"), b=("retiro", "sum"))
    rat = (d.a / d.b).values
    ci[k] = bootstrap_ci(rat, np.mean, seed=1)
tab["ic95_razon_por_dia"] = pd.Series(ci)
tab.to_csv(out2 / "salida_no_programa_por_contexto.csv")
res["EV-02"]["salida_no_programa"] = tab.reset_index().to_dict("records")
print("EV-02 estados:\n", est.round(4), "\n\nSalida no-programa (MON/MOV, días sin traspaso):\n", tab.round(4))

# ------------------------------------------------------------------ EV-03
out3 = salida("EV-03")
mv, mon = fuente("movimientos_diario"), fuente("monitoreo_cortes")
dia = mov_diario(mv)
par = detectar_parciales(dia, mon[mon.hora_corte == 1700])
par["mes"] = par.id_fecha // 100
mens = par.groupby("mes").agg(r_mediana_suc=("r_mediana", "median"), pct_dias_parcial=("parcial", "mean"))
prog = mv[mv.monto > 0].groupby(mv.id_fecha // 100).id_programa.nunique().rename("n_programas")
mens = mens.join(prog)
# cobertura del total no-traspaso: razón de sumas MOV / MON@1700 en sucursal-días sin traspaso grande
pp = p[(p.retiro.notna()) & (p.retiros_1700 > 0) & (p.n_trans > 0)].copy()
pp["mes"] = pp.id_fecha // 100
pp["ok"] = (pp.retiros_1700 - pp.retiro) <= np.maximum(0.05 * pp.retiro, 50000)
mens["pct_suc_dias_concilian"] = pp.groupby("mes").ok.mean()
mens["razon_suma_mov_mon"] = pp.groupby("mes").retiro.sum() / pp.groupby("mes").retiros_1700.sum()
mens.to_csv(out3 / "cobertura_mensual.csv")
print("\nEV-03 cobertura mensual:\n", mens.round(3).to_string())

# Quiebre: primer mes a partir del cual la sucursal mediana concilia (r >= 0.98) en TODOS los meses siguientes
ok = mens.r_mediana_suc >= 0.98
inicio = None
for m in mens.index:
    if ok.loc[m:].all():
        inicio = int(m)
        break
res["EV-03"] = {"primer_mes_estable": inicio, "mensual": mens.reset_index().to_dict("records")}
guardar_json("EV-02", res["EV-02"])
guardar_json("EV-03", res["EV-03"])
print("\nPrimer mes desde el que la sucursal mediana concilia siempre:", inicio)
