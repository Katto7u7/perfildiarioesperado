"""EV-04 Validez y estabilidad de la forma intradía  |  EV-05 Supuesto de apertura 08:00.

EV-04 preguntas:
 a) ¿la forma de MONITOREO (salida total) representa la del retiro de clientes? -> MOVIMIENTOS_CORTE 13:30 (2026-11-13)
 b) ¿el periodo con carga retroactiva (backfill) tiene la misma forma que el periodo de carga nativa diaria?
 c) ¿la forma es una característica estable de cada sucursal (varianza entre sucursales >> dentro)?
 d) ¿la forma cambia por contexto (día de dispersión, día de la semana, régimen, quincena)?
EV-05: sensibilidad de velocidad pico / agresividad a la hora de apertura (07:30, 08:00, 08:30).
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from comun import CFG, bootstrap_ci, guardar_json, panel_completo, salida

out4, out5 = salida("EV-04"), salida("EV-05")
p = panel_completo()
f = p[(p.id_fecha >= 20250602) & p.forma_valida].copy()
res = {"n_sucursal_dias_forma_valida": int(len(f)), "n_sucursales": int(f.id_sucursal.nunique())}

# ---------------------------------------------------------------- a) MOVIMIENTOS_CORTE 13:30
con = sqlite3.connect("file:" + str(CFG.ruta("db_saldos")).replace("\\", "/") + "?mode=ro", uri=True)
mc = pd.read_sql("select id_sucursal, sum(monto) m1330 from MOVIMIENTOS_CORTE where id_fecha=20261113 and hora_corte=1330 group by 1", con)
lag = pd.read_sql("select id_fecha, substr(created_at_utc,1,10) c, count(*) n from MONITOREO_CORTE "
                  "where id_fecha>=20250602 and hora_corte=1700 group by 1,2", con)
con.close()
d13 = p[p.id_fecha == 20261113][["id_sucursal", "retiro", "retiros_1300", "retiros_1700", "forma_valida"]].merge(mc, on="id_sucursal")
d13 = d13[d13.forma_valida & (d13.retiro > 0)]
d13["s_mov_1330"] = d13.m1330 / d13.retiro
d13["s_mon_1300"] = d13.retiros_1300 / d13.retiros_1700
rho = stats.spearmanr(d13.s_mov_1330, d13.s_mon_1300)
res["a_movimientos_corte"] = {
    "n_suc": int(len(d13)), "spearman": float(rho.statistic),
    "mediana_s_mov_1330": float(d13.s_mov_1330.median()), "mediana_s_mon_1300": float(d13.s_mon_1300.median()),
    "pct_mov1330_ge_mon1300": float((d13.s_mov_1330 >= d13.s_mon_1300 - 0.01).mean()),
    "nota": "13:30 debe ser >= 13:00 si ambas fuentes describen el mismo proceso"}

# ---------------------------------------------------------------- b) backfill vs nativo
lag["c"] = pd.to_datetime(lag.c)
lag["f"] = pd.to_datetime(lag.id_fecha.astype(str), format="%Y%m%d")
lag["dias"] = (lag.c - lag.f).dt.days
dom = lag.sort_values("n").groupby("id_fecha").tail(1)            # creación predominante por fecha
nativas = set(dom[dom.dias.between(0, 3)].id_fecha)
f["nativo"] = f.id_fecha.isin(nativas)
res["b_fechas_nativas"] = {"n": len(nativas), "rango": [int(min(nativas)), int(max(nativas))] if nativas else None}
# comparar mismo mes calendario-régimen sería ideal; usamos todos los días y, aparte, solo días sin dispersión
comp = []
for c in (900, 1100, 1300):
    for sub, mask in (("todos", np.ones(len(f), bool)), ("sin_dispersion", ~f.disp_alguna.fillna(False).values)):
        a, b = f.loc[mask & f.nativo, f"p_{c}"], f.loc[mask & ~f.nativo, f"p_{c}"]
        ks = stats.ks_2samp(a, b)
        comp.append({"corte": c, "subconjunto": sub, "n_nativo": len(a), "n_backfill": len(b),
                     "mediana_nativo": a.median(), "mediana_backfill": b.median(), "ks_D": ks.statistic})
comp = pd.DataFrame(comp)
# dispersión ENTRE sucursales (si el backfill fuera sintético con curva fija, sería ~0)
disp_suc = f.groupby(["nativo", "id_sucursal"]).p_1100.median().groupby("nativo").agg(["std", "count"])
res["b_backfill_vs_nativo"] = comp.to_dict("records")
res["b_std_entre_sucursales_p1100"] = disp_suc.reset_index().to_dict("records")
comp.to_csv(out4 / "backfill_vs_nativo.csv", index=False)

# ---------------------------------------------------------------- c) estabilidad por sucursal (ICC)
icc = {}
for c in (900, 1100, 1300):
    g = f.groupby("id_sucursal")[f"p_{c}"]
    k = g.size()
    keep = k[k >= 30].index
    x = f[f.id_sucursal.isin(keep)]
    gm = x.groupby("id_sucursal")[f"p_{c}"].transform("mean")
    ssw = ((x[f"p_{c}"] - gm) ** 2).sum()
    ssb = ((gm - x[f"p_{c}"].mean()) ** 2).sum()
    icc[c] = float(ssb / (ssb + ssw))                      # proporción de varianza entre sucursales (eta²)
# estabilidad temporal: forma mediana por sucursal primera mitad vs segunda mitad
mid = 20260201
h1 = f[f.id_fecha < mid].groupby("id_sucursal").p_1100.median()
h2 = f[f.id_fecha >= mid].groupby("id_sucursal").p_1100.median()
j = pd.concat([h1, h2], axis=1, keys=["h1", "h2"]).dropna()
res["c_eta2_entre_sucursales"] = icc
res["c_estabilidad_mitades_p1100"] = {"spearman": float(stats.spearmanr(j.h1, j.h2).statistic), "n": int(len(j)),
                                       "mad_cambio": float((j.h1 - j.h2).abs().median())}

# ---------------------------------------------------------------- d) efecto del contexto sobre la forma
f["ctx_disp"] = np.select([f.disp_am.fillna(False), f.disp_becas.fillna(False)], ["AM", "BECAS"], default="SIN")
ctx_rows = []
for col in ("ctx_disp", "dow", "regimen", "es_quincena", "post_cierre"):
    # efecto dentro de sucursal: desviación de p_c respecto a la mediana propia de la sucursal
    for c in (900, 1100, 1300):
        dev = f[f"p_{c}"] - f.groupby("id_sucursal")[f"p_{c}"].transform("median")
        t = dev.groupby(f[col]).agg(["median", "mean", "size"])
        for lvl, r in t.iterrows():
            ctx_rows.append({"factor": col, "nivel": str(lvl), "corte": c, "desv_mediana_vs_propia": r["median"],
                             "desv_media": r["mean"], "n": int(r["size"])})
ctx = pd.DataFrame(ctx_rows)
ctx.to_csv(out4 / "efecto_contexto_forma.csv", index=False)
# forma agregada
agg = f[[f"p_{c}" for c in (900, 1100, 1300)]].median()
res["forma_mediana_global"] = agg.to_dict()
guardar_json("EV-04", res)

# ---------------------------------------------------------------- EV-05 apertura
r5 = []
med = f.groupby("id_sucursal")[["p_900", "p_1100", "p_1300"]].median()
med["p_1700"] = 1.0
for ap in (730, 800, 830):
    h0 = (ap // 100) + (ap % 100) / 60
    horas = np.array([9 - h0, 2, 2, 4])
    shares = np.column_stack([med.p_900, med.p_1100 - med.p_900, med.p_1300 - med.p_1100, 1 - med.p_1300])
    vel_rel = shares / horas                              # fracción del día por hora
    pico = np.array([900, 1100, 1300, 1700])[vel_rel.argmax(axis=1)]
    agres = med.p_1100 / ((11 - h0) / (17 - h0))
    r5.append({"apertura": ap, "pct_pico_primer_intervalo": float((pico == 900).mean()),
               "pct_pico_9_11": float((pico == 1100).mean()), "pct_pico_11_13": float((pico == 1300).mean()),
               "pct_pico_13_17": float((pico == 1700).mean()), "agresividad_mediana": float(agres.median()),
               "pico": pico})
base = r5[1]["pico"]
for r in r5:
    r["pct_sucursales_mismo_pico_que_0800"] = float((r["pico"] == base).mean())
    del r["pico"]
r5 = pd.DataFrame(r5)
r5.to_csv(out5 / "sensibilidad_apertura.csv", index=False)
guardar_json("EV-05", {"sensibilidad": r5.to_dict("records")})

pd.set_option("display.width", 220)
for k, v in res.items():
    if k != "b_backfill_vs_nativo":
        print(k, v)
print(comp.round(4).to_string())
print(ctx.pivot_table(index=["factor", "nivel"], columns="corte", values="desv_mediana_vs_propia").round(4).to_string())
print(r5.round(3).to_string())
