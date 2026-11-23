"""EV-08 ¿PROGRAMAS_EMERGENTES (LEEN/LCEN por sucursal) aporta información predictiva sin fuga?

Disponibilidad: solo eventos con id_fecha_solicitud <= fecha de corte (se mide el adelanto solicitud->evento).
Pruebas:
 1) ¿Qué fracción del retiro LEEN/LCEN ocurre dentro de ventanas de evento?
 2) ¿El monto planeado por sucursal predice el retiro LEEN realizado en la ventana? (Spearman, razón)
 3) ¿Los días de evento tienen retiro total por encima de lo que el calendario explica? (residuo vs efecto fijo)
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from comun import CFG, fuente, guardar_json, panel_completo, salida

out = salida("EV-08")
con = sqlite3.connect("file:" + str(CFG.ruta("db_operativo")).replace("\\", "/") + "?mode=ro", uri=True)
pe = pd.read_sql("select * from PROGRAMAS_EMERGENTES", con)
con.close()
pe["f_sol"] = pd.to_datetime(pe.id_fecha_solicitud.astype(str))
pe["f_ev"] = pd.to_datetime(pe.id_fecha_evento.astype(str))
pe["adelanto_dias"] = (pe.f_ev - pe.f_sol).dt.days
res = {"filas": len(pe), "programas": pe.id_programa.value_counts().to_dict(),
       "eventos": int(pe.evento_codigo.nunique()), "sucursales": int(pe.id_sucursal.nunique()),
       "rango_evento": [int(pe.id_fecha_evento.min()), int(pe.id_fecha_evento.max())],
       "estatus": pe.estatus_evento.value_counts().to_dict(),
       "adelanto_solicitud_evento_dias": pe.adelanto_dias.describe().round(1).to_dict()}

# ventana de evento por sucursal: [fecha_evento, fecha_evento + días naturales del código]
cod = pe.evento_codigo.str.extract(r"_(\d{8})_(\d{8})")
pe["v_ini"] = pd.to_numeric(cod[0]).fillna(pe.id_fecha_evento)
pe["v_fin"] = pd.to_numeric(cod[1]).fillna(pe.id_fecha_evento)

mv = fuente("movimientos_diario")
prog = mv[mv.id_programa.isin([6, 7]) & (mv.id_fecha >= 20260601)]
# asignar cada retiro LEEN/LCEN a evento de su sucursal
m = prog.merge(pe[["id_sucursal", "id_programa", "evento_codigo", "v_ini", "v_fin", "total_monto_pe"]],
               on=["id_sucursal", "id_programa"], how="left")
m["en_ventana"] = (m.id_fecha >= m.v_ini) & (m.id_fecha <= m.v_fin)
dentro = m[m.en_ventana].drop_duplicates(["id_fecha", "id_sucursal", "id_programa"])
tot = prog[prog.id_fecha <= int(pe.v_fin.max())].monto.sum()
res["pct_retiro_LEEN_LCEN_en_ventana_evento"] = float(dentro.monto.sum() / tot) if tot else None

# 2) monto planeado vs realizado en la ventana (por sucursal-evento)
real = dentro.groupby(["evento_codigo", "id_sucursal"]).monto.sum().rename("real")
plan = pe.groupby(["evento_codigo", "id_sucursal"]).agg(plan=("total_monto_pe", "max")).join(real, how="left").fillna({"real": 0})
plan = plan[plan.plan > 0]
rho = stats.spearmanr(plan.plan, plan.real)
res["plan_vs_real"] = {"n": int(len(plan)), "spearman": float(rho.statistic),
                       "razon_suma_real_plan": float(plan.real.sum() / plan.plan.sum()),
                       "mediana_razon_por_sucursal": float((plan.real / plan.plan).median())}

# 3) residuo del retiro TOTAL en días de evento vs días comparables sin evento (misma sucursal)
p = panel_completo()
v = p[(p.id_fecha >= 20260301) & p.valido].copy()
v["y"] = np.log1p(v.retiro)
# efecto fecha (calendario) removido: y - media sucursal - efecto fecha
v["y_dm"] = v.y - v.groupby("id_sucursal").y.transform("mean")
v["resid"] = v.y_dm - v.groupby("id_fecha").y_dm.transform("mean")
ev_days = pe[["id_sucursal", "v_ini", "v_fin", "total_monto_pe", "f_sol"]]
v = v.merge(ev_days, on="id_sucursal", how="left")
v["evento"] = (v.id_fecha >= v.v_ini) & (v.id_fecha <= v.v_fin)
vv = v.sort_values("evento", ascending=False).drop_duplicates(["id_sucursal", "id_fecha"])
a, b = vv[vv.evento].resid, vv[~vv.evento & vv.v_ini.notna()].resid
res["residuo_dias_evento"] = {"n_evento": int(len(a)), "media_resid_evento": float(a.mean()),
                              "media_resid_no_evento": float(b.mean()),
                              "efecto_pct": float((np.exp(a.mean() - b.mean()) - 1) * 100),
                              "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue)}
# dosis-respuesta: ¿el residuo crece con el monto planeado relativo al retiro típico de la sucursal?
tip = p[(p.id_fecha >= 20250602) & p.valido].groupby("id_sucursal").retiro.median().rename("tipico")
e = vv[vv.evento].merge(tip, on="id_sucursal")
e["rel"] = np.log1p(e.total_monto_pe / e.tipico.clip(lower=1))
rho2 = stats.spearmanr(e.rel, e.resid)
res["dosis_respuesta_spearman"] = float(rho2.statistic)
guardar_json("EV-08", res)
for k, val in res.items():
    print(k, ":", val)
