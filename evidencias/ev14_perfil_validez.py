"""EV-14 Validez de los indicadores del perfil y calibración de tipo_perfil.

Para cada corte mensual t (último día operativo de cada mes 2026-01..2026-09) se calcula el perfil con datos <= t y se
mide lo REALIZADO en los 20 días operativos siguientes (t+1..t+20):
  - tasa de días críticos (saldo mínimo < P90 causal), tasa de quiebre intradía, volatilidad residual,
    frecuencia de días extremos, intensidad AM, cobertura realizada.
Pruebas: (1) validez predictiva de cada indicador (Spearman indicador_t vs realizado_t+), (2) redundancia (Spearman
entre indicadores), (3) umbrales τ: lift de la categoría sobre la tasa global, (4) estabilidad (kappa entre cortes),
(5) estructura: comparación con clustering GMM (ARI).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from comun import CFG, calendario_op, fuente, guardar_json, panel_completo, salida
from perfil_retiro.modelo import contexto_b1, variables
from perfil_retiro.perfil import calcular_perfil

out = salida("EV-14")
p = panel_completo()
p = p[p.id_fecha >= 20250602]
cal = calendario_op()
suc = fuente("sucursal")
ops = cal[cal.operativo].id_fecha.sort_values().values
cortes = []
for ym in range(202601, 202610):
    m = ops[(ops // 100) == ym]
    if len(m):
        cortes.append(int(m[-1]))

IND = ["retiro_promedio_diario", "volatilidad_retiro_diario", "dias_cobertura_promedio", "dias_cobertura_p90",
       "pct_dias_saldo_critico", "pct_quiebre_intradia", "velocidad_consumo_promedio", "velocidad_consumo_p90",
       "aceleracion_abs_p90", "indice_agresividad_consumo", "indice_concentracion_intradia", "participacion_intervalo_pico",
       "pct_dias_consumo_extremo", "score_riesgo_desabasto", "score_volatilidad", "intensidad_dispersion"]

perfiles, realizados = [], []
v = p[p.valido].sort_values(["id_sucursal", "id_fecha"]).copy()
g = v.groupby(["id_sucursal", "regimen"]).retiro
v["p90_prev"] = g.transform(lambda s: s.rolling(60, min_periods=20).quantile(0.9).shift(1))
v["p50_hoy"] = g.transform(lambda s: s.rolling(60, min_periods=20).median())
v["p90_hoy"] = g.transform(lambda s: s.rolling(60, min_periods=20).quantile(0.9))
fmed_all = v[v.forma_valida].groupby("id_sucursal")[["p_900", "p_1100", "p_1300"]].median()
X = variables(v)
v["ctx"] = contexto_b1(X).str.split("_").str[0].values
v["ly"] = np.log1p(v.retiro)
for t in cortes:
    pf, _, _, aux = calcular_perfil(p[p.id_fecha <= t], cal, suc, CFG, t)
    pf["corte"] = t
    perfiles.append(pf)
    fut_d = ops[ops > t][:20]
    f = v[v.id_fecha.isin(fut_d)].copy()
    # mediana de contexto con historia <= t (para residuos realizados sin usar el futuro en la referencia)
    ref = v[(v.id_fecha <= t)].groupby(["id_sucursal", "ctx"]).ly.median().rename("ref")
    f = f.join(ref, on=["id_sucursal", "ctx"])
    f["res"] = f.ly - f.ref
    fs = f[f.saldo_valido & f.p90_prev.notna() & f.p90_hoy.notna()].copy()
    fm = fmed_all.reindex(fs.id_sucursal).fillna(fmed_all.median()).values      # forma típica (aprox. causal)
    qb = np.zeros(len(fs), bool)
    for j, c in enumerate((900, 1100, 1300)):
        qb |= fs[f"sdo_final_{c}"].values < (1 - fm[:, j]) * fs.p90_prev.values
    qb |= fs.sdo_final_1700.values < 0
    fs["quiebre"], fs["critico"] = qb, fs.saldo_cierre < fs.p90_hoy
    r = pd.DataFrame({
        "real_tasa_critico": fs.groupby("id_sucursal").critico.mean(),
        "real_tasa_quiebre": fs.groupby("id_sucursal").quiebre.mean(),
        "real_tasa_riesgo": (fs.critico | fs.quiebre).groupby(fs.id_sucursal).mean(),
        "real_cobertura": (fs.saldo_cierre / fs.p50_hoy.where(fs.p50_hoy > 0)).clip(upper=30).groupby(fs.id_sucursal).mean(),
        "real_vol_residual": f.groupby("id_sucursal").res.apply(lambda s: 1.4826 * (s - s.median()).abs().median()),
        "real_extremos": f.groupby("id_sucursal").res.apply(lambda s: ((s - s.median()) > 3 * 1.4826 * (s - s.median()).abs().median()).mean()),
        "real_retiro_medio": f.groupby("id_sucursal").retiro.mean(),
    })
    am1 = f[f.ctx == "AM1"].groupby("id_sucursal").retiro.median()
    sn = f[f.ctx != "AM1"].groupby("id_sucursal").retiro.median()
    r["real_intensidad_am"] = am1 / sn.where(sn > 0)
    fz = f[f.forma_valida]
    r["real_agresividad"] = fz.groupby("id_sucursal").p_1100.median()
    r["corte"] = t
    realizados.append(r.reset_index())
P = pd.concat(perfiles, ignore_index=True)
R = pd.concat(realizados, ignore_index=True)
J = P.merge(R, on=["id_sucursal", "corte"], how="inner")
J.to_parquet(out / "perfiles_vs_realizado.parquet", index=False)

# (1) validez predictiva: indicador -> su contraparte realizada
pares = {"pct_dias_saldo_critico": "real_tasa_critico", "score_riesgo_desabasto": "real_tasa_riesgo",
         "pct_quiebre_intradia": "real_tasa_quiebre", "dias_cobertura_promedio": "real_cobertura",
         "dias_cobertura_p90": "real_cobertura", "score_volatilidad": "real_vol_residual",
         "volatilidad_retiro_diario": "real_vol_residual", "pct_dias_consumo_extremo": "real_extremos",
         "retiro_promedio_diario": "real_retiro_medio", "intensidad_dispersion": "real_intensidad_am",
         "indice_agresividad_consumo": "real_agresividad", "indice_concentracion_intradia": "real_agresividad",
         "velocidad_consumo_promedio": "real_retiro_medio", "velocidad_consumo_p90": "real_retiro_medio",
         "aceleracion_abs_p90": "real_vol_residual", "participacion_intervalo_pico": "real_agresividad"}
val = []
for ind, real in pares.items():
    x = J[[ind, real, "corte"]].dropna()
    rho = stats.spearmanr(x[ind], x[real]).statistic
    rho_c = x.groupby("corte").apply(lambda t: stats.spearmanr(t[ind], t[real]).statistic, include_groups=False)
    val.append({"indicador": ind, "realizado_t+20": real, "spearman": rho, "spearman_min_por_corte": rho_c.min(),
                "spearman_max_por_corte": rho_c.max(), "n": len(x)})
val = pd.DataFrame(val).sort_values("spearman", ascending=False)
val.to_csv(out / "validez_predictiva.csv", index=False)

# (2) redundancia
red = P[IND].corr(method="spearman")
red.to_csv(out / "redundancia_spearman.csv")
pares_red = [(a, b, red.loc[a, b]) for i, a in enumerate(IND) for b in IND[i + 1:] if abs(red.loc[a, b]) > 0.85]

# (3) calibración de umbrales por lift sobre la tasa global realizada
def lift_curve(ind, real, grid):
    base = J[real].mean()
    rows = []
    for tau in grid:
        sel = J[J[ind] >= tau]
        rows.append({"tau": tau, "pct_sucursales": len(sel) / len(J), "tasa_real_sel": sel[real].mean(),
                     "lift": sel[real].mean() / base if base else np.nan})
    return pd.DataFrame(rows)


curvas = {
    "ALTO_RIESGO": ("score_riesgo_desabasto", "real_tasa_riesgo", np.round(np.arange(0.04, 0.41, 0.02), 2)),
    "QUIEBRE_INTRADIA": ("pct_quiebre_intradia", "real_tasa_quiebre", np.round(np.arange(0.01, 0.21, 0.01), 2)),
    "VOLATIL": ("score_volatilidad", "real_vol_residual", np.round(np.arange(0.3, 1.21, 0.1), 2)),
    "DISPERSION": ("intensidad_dispersion", "real_intensidad_am", np.round(np.arange(3, 13, 1.0), 1)),
}
cal_tau = {}
for nom, (ind, real, grid) in curvas.items():
    c = lift_curve(ind, real, grid)
    c.to_csv(out / f"lift_{nom}.csv", index=False)
    ok = c[(c.lift >= 2.0) & (c.pct_sucursales >= 0.03)] if nom != "DISPERSION" else c[(c.lift >= 1.5) & (c.pct_sucursales >= 0.05)]
    cal_tau[nom] = {"tau_elegido": float(ok.tau.min()) if len(ok) else None, "curva": c.round(3).to_dict("records")}
# SOBREDOTADA: cobertura alta que PERSISTE y riesgo bajo
c = []
for tau in [6, 8, 10, 12, 15, 20]:
    sel = J[(J.dias_cobertura_promedio >= tau) & (J.score_riesgo_desabasto < 0.05)]
    c.append({"tau": tau, "pct_sucursales": len(sel) / len(J), "cobertura_real_mediana": sel.real_cobertura.median(),
              "pct_persiste_>=tau*0.7": (sel.real_cobertura >= 0.7 * tau).mean(), "tasa_critico_real": sel.real_tasa_critico.mean()})
c = pd.DataFrame(c)
c.to_csv(out / "lift_SOBREDOTADA.csv", index=False)
okc = c[(c["pct_persiste_>=tau*0.7"] >= 0.8) & (c.tasa_critico_real <= J.real_tasa_critico.mean() / 3)]
cal_tau["SOBREDOTADA"] = {"tau_elegido": float(okc.tau.min()) if len(okc) else None, "curva": c.round(3).to_dict("records")}

# (4) estabilidad de tipo_perfil entre cortes consecutivos (con umbrales actuales de config)
kap = []
for a, b in zip(cortes[:-1], cortes[1:]):
    x = P[P.corte == a][["id_sucursal", "tipo_perfil"]].merge(P[P.corte == b][["id_sucursal", "tipo_perfil"]], on="id_sucursal")
    kap.append({"de": a, "a": b, "kappa": cohen_kappa_score(x.tipo_perfil_x, x.tipo_perfil_y),
                "pct_igual": float((x.tipo_perfil_x == x.tipo_perfil_y).mean())})
for a, b in zip(cortes[:-2], cortes[2:]):                     # mismo régimen (t y t+2 meses)
    x = P[P.corte == a][["id_sucursal", "tipo_perfil"]].merge(P[P.corte == b][["id_sucursal", "tipo_perfil"]], on="id_sucursal")
    kap.append({"de": a, "a": b, "kappa": cohen_kappa_score(x.tipo_perfil_x, x.tipo_perfil_y),
                "pct_igual": float((x.tipo_perfil_x == x.tipo_perfil_y).mean()), "mismo_regimen": True})
kap = pd.DataFrame(kap)
dist = P.groupby(["corte", "tipo_perfil"]).size().unstack(fill_value=0)
dist.to_csv(out / "distribucion_tipo_perfil.csv")
print(dist.to_string())
# validez de la categoría: tasa crítica / volatilidad realizadas por tipo
vt = J.groupby("tipo_perfil")[["real_tasa_riesgo", "real_tasa_critico", "real_tasa_quiebre", "real_cobertura", "real_vol_residual", "real_intensidad_am"]].median()
vt["n"] = J.groupby("tipo_perfil").size()
vt.to_csv(out / "realizado_por_tipo.csv")
print(vt.round(3).to_string())
kw = stats.kruskal(*[g.real_tasa_riesgo.dropna() for _, g in J.groupby("tipo_perfil")])
print("Kruskal-Wallis tasa crítica realizada por tipo: H=%.1f p=%.2e" % (kw.statistic, kw.pvalue))

# (5) estructura: GMM sobre indicadores estandarizados vs reglas
Z = P[P.corte == cortes[-1]].dropna(subset=IND)
Xs = StandardScaler().fit_transform(np.log1p(Z[IND].clip(lower=0)))
gm = GaussianMixture(6, random_state=0, n_init=3).fit(Xs)
ari = adjusted_rand_score(Z.tipo_perfil, gm.predict(Xs))
res = {"cortes": cortes, "n_filas": int(len(J)), "validez_predictiva": val.round(3).to_dict("records"),
       "pares_redundantes_|rho|>0.85": pares_red, "calibracion_tau": cal_tau, "estabilidad_kappa": kap.round(3).to_dict("records"),
       "ARI_reglas_vs_GMM6": float(ari), "kruskal_tasa_critica": [float(kw.statistic), float(kw.pvalue)],
       "realizado_por_tipo": vt.round(4).reset_index().to_dict("records"), "tasa_base_critico": float(J.real_tasa_critico.mean()), "tasa_base_riesgo": float(J.real_tasa_riesgo.mean()),
       "tasa_base_quiebre": float(J.real_tasa_quiebre.mean())}
guardar_json("EV-14", res)
pd.set_option("display.width", 220)
print(val.round(3).to_string())
print("\nredundantes:", [(a, b, round(r, 3)) for a, b, r in pares_red])
for k, vv in cal_tau.items():
    print("\n", k, "tau =", vv["tau_elegido"])
    print(pd.DataFrame(vv["curva"]).to_string())
print("\nkappa:\n", kap.round(3).to_string(), "\nARI reglas vs GMM:", round(ari, 3))
