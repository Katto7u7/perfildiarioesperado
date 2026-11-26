"""EV-06 Efectos de calendario | EV-07 Heterogeneidad entre sucursales | EV-09 Régimen IMPAR/PAR.

Diseño de inferencia honesto: los efectos de calendario varían por FECHA, no por sucursal. Por eso:
  1) efecto_fecha(d) = media sobre sucursales de [log1p(retiro) - media propia de la sucursal]  (efecto fijo de sucursal)
  2) OLS de efecto_fecha sobre variables de calendario con errores HAC (Newey-West), n = nº de fechas.
Así no se infla la significancia con 1 millón de filas correlacionadas.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from comun import guardar_json, panel_completo, salida

out6, out7, out9 = salida("EV-06"), salida("EV-07"), salida("EV-09")
p = panel_completo()
v = p[(p.id_fecha >= 20250602) & p.valido].copy()
v["y"] = np.log1p(v.retiro)
# sucursales con historia suficiente
k = v.groupby("id_sucursal").size()
v = v[v.id_sucursal.isin(k[k >= 120].index)]
v["y_dm"] = v.y - v.groupby("id_sucursal").y.transform("mean")
fe = v.groupby("id_fecha").agg(ef=("y_dm", "mean"), n=("y_dm", "size")).reset_index()
cal_cols = ["dow", "regimen", "es_quincena", "dias_al_fin_mes", "dia_habil_mes", "post_cierre", "post_feriado",
            "pre_feriado", "pos_quincena", "disp_am", "disp_am_segundo", "disp_am_primero", "disp_am_retiro_ref",
            "disp_becas", "disp_becas_segundo", "disp_becas_primero", "disp_otro", "mes", "anio"]
fe = fe.merge(v.drop_duplicates("id_fecha")[["id_fecha"] + cal_cols], on="id_fecha")
fe = fe.sort_values("id_fecha").reset_index(drop=True)


def disenio(df, con_regimen=True, con_monto=False):
    X = pd.DataFrame(index=df.index)
    for d, nom in ((1, "martes"), (2, "miercoles"), (3, "jueves"), (4, "viernes")):
        X[nom] = (df.dow == d).astype(float)
    X["am_dia1"] = (df.disp_am & df.disp_am_primero).astype(float)
    X["am_dia2"] = (df.disp_am & df.disp_am_segundo & ~df.disp_am_primero).astype(float)
    X["becas_dia1"] = (df.disp_becas & df.disp_becas_primero).astype(float)
    X["becas_dia2"] = (df.disp_becas & df.disp_becas_segundo & ~df.disp_becas_primero).astype(float)
    X["otro_programa"] = df.disp_otro.astype(float)
    X["quincena_15"] = (df.es_quincena & (df.dias_al_fin_mes > 0)).astype(float)
    X["fin_de_mes"] = (df.dias_al_fin_mes == 0).astype(float)
    X["post_quincena"] = df.pos_quincena.astype(float)
    X["inicio_mes_d1_3"] = (df.dia_habil_mes <= 3).astype(float)
    X["post_feriado"] = df.post_feriado.astype(float)
    X["pre_feriado"] = df.pre_feriado.astype(float)
    X["diciembre"] = (df.mes == 12).astype(float)
    if con_regimen:
        X["regimen_impar"] = (df.regimen == "IMPAR").astype(float)
    if con_monto:
        m = np.log(df.disp_am_retiro_ref.where(df.disp_am_retiro_ref > 0))
        X["log_monto_am_c"] = (m - m.mean()).fillna(0.0)
    t = (pd.to_datetime(df.id_fecha.astype(str)) - pd.Timestamp("2025-06-02")).dt.days / 365.25
    X["tendencia_anual"] = t
    return sm.add_constant(X)


def ajustar(df, **kw):
    X = disenio(df, **kw)
    m = sm.OLS(df.ef, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    t = pd.DataFrame({"coef_log": m.params, "se_hac": m.bse, "p": m.pvalues})
    t["efecto_pct"] = (np.exp(t.coef_log) - 1) * 100
    t["ic95_pct"] = [f"[{(np.exp(a)-1)*100:.1f}, {(np.exp(b)-1)*100:.1f}]" for a, b in m.conf_int().values]
    return m, t


m_full, t_full = ajustar(fe, con_regimen=True, con_monto=True)
t_full.to_csv(out6 / "efectos_calendario.csv")
print("EV-06  R2 =", round(m_full.rsquared, 3), " n fechas =", int(m_full.nobs))
print(t_full.round(4).to_string())

# Ablación a nivel fecha: R2 de cada bloque
bloques = {
    "dia_semana": ["martes", "miercoles", "jueves", "viernes"],
    "dispersion_AM": ["am_dia1", "am_dia2", "log_monto_am_c"],
    "dispersion_BECAS": ["becas_dia1", "becas_dia2"],
    "otro_programa": ["otro_programa"],
    "quincena_fin_mes": ["quincena_15", "fin_de_mes", "post_quincena", "inicio_mes_d1_3"],
    "feriados": ["post_feriado", "pre_feriado"],
    "diciembre": ["diciembre"], "regimen": ["regimen_impar"], "tendencia": ["tendencia_anual"],
}
Xf = disenio(fe, True, True)
abl = []
for b, cols in bloques.items():
    Xr = Xf.drop(columns=cols)
    r2 = sm.OLS(fe.ef, Xr).fit().rsquared
    abl.append({"bloque": b, "R2_sin_bloque": r2, "perdida_R2": m_full.rsquared - r2})
abl = pd.DataFrame(abl).sort_values("perdida_R2", ascending=False)
abl.to_csv(out6 / "ablacion_bloques.csv", index=False)
print("\nAblación (pérdida de R2 al quitar cada bloque):\n", abl.round(4).to_string())

# ------------------------------------------------------------------ EV-09 régimen
# ¿el régimen cambia el nivel en días SIN dispersión de AM/BECAS?
sin = fe[~fe.disp_am & ~fe.disp_becas]
m9, t9 = ajustar(sin, con_regimen=True, con_monto=False)
t9 = t9.drop(index=[i for i in ["am_dia1", "am_dia2", "becas_dia1", "becas_dia2"] if i in t9.index], errors="ignore")
reg_sin = t9.loc["regimen_impar"]
# razón cruda de retiro medio por sucursal-día impar/par (para contraste con el 2.28 de la política)
v["regimen"] = v.regimen
raw = v.groupby("regimen").retiro.mean()
raw_sin = v[~v.disp_am.astype(bool) & ~v.disp_becas.astype(bool)].groupby("regimen").retiro.mean()
raw_am = v[v.disp_am.astype(bool)].retiro.mean() / v[~v.disp_am.astype(bool) & ~v.disp_becas.astype(bool)].retiro.mean()
res9 = {
    "razon_cruda_impar_par_todos": float(raw["IMPAR"] / raw["PAR"]),
    "razon_cruda_impar_par_dias_sin_dispersion": float(raw_sin["IMPAR"] / raw_sin["PAR"]),
    "razon_cruda_dia_AM_vs_sin_dispersion": float(raw_am),
    "efecto_regimen_en_dias_sin_dispersion_pct": float(reg_sin.efecto_pct), "p": float(reg_sin.p),
    "ic95": reg_sin.ic95_pct, "n_fechas_sin_dispersion": int(len(sin)),
    "efecto_regimen_condicional_modelo_completo_pct": float(t_full.loc["regimen_impar", "efecto_pct"]),
    "p_condicional": float(t_full.loc["regimen_impar", "p"]),
}
print("\nEV-09:", res9)

# ------------------------------------------------------------------ EV-07 heterogeneidad (DerSimonian-Laird)
def efecto_suc(mask_trat, mask_ctrl, nombre):
    """Efecto por sucursal: diferencia de medias de log1p(retiro) tratamiento vs control (mismo periodo)."""
    g1 = v[mask_trat].groupby("id_sucursal").y.agg(["mean", "var", "size"])
    g0 = v[mask_ctrl].groupby("id_sucursal").y.agg(["mean", "var", "size"])
    j = g1.join(g0, lsuffix="1", rsuffix="0").dropna()
    j = j[(j.size1 >= 10) & (j.size0 >= 10)]
    eff = j.mean1 - j.mean0
    var = j.var1 / j.size1 + j.var0 / j.size0
    ok = np.isfinite(var) & (var > 1e-6)
    eff, var = eff[ok], var[ok]
    w = 1 / var
    mu_fe = (w * eff).sum() / w.sum()
    Q = (w * (eff - mu_fe) ** 2).sum()
    dfq = len(eff) - 1
    tau2 = max(0.0, (Q - dfq) / (w.sum() - (w ** 2).sum() / w.sum()))
    I2 = max(0.0, (Q - dfq) / Q)
    return {"efecto": nombre, "n_suc": int(len(eff)), "media_pct": float((np.exp(eff.mean()) - 1) * 100),
            "p10_pct": float((np.exp(eff.quantile(.1)) - 1) * 100), "p90_pct": float((np.exp(eff.quantile(.9)) - 1) * 100),
            "tau_log": float(np.sqrt(tau2)), "I2": float(I2),
            "se_mediano_log": float(np.sqrt(var).median())}, eff


sin_disp = ~v.disp_am.astype(bool) & ~v.disp_becas.astype(bool) & ~v.disp_otro.astype(bool)
het = []
r, eff_am = efecto_suc(v.disp_am.astype(bool) & v.disp_am_primero.astype(bool), sin_disp, "dia_AM_1 vs sin_dispersion")
het.append(r)
r, _ = efecto_suc(v.disp_becas.astype(bool) & v.disp_becas_primero.astype(bool), sin_disp, "dia_BECAS_1 vs sin_dispersion")
het.append(r)
r, _ = efecto_suc(sin_disp & (v.dow == 0), sin_disp & (v.dow.between(1, 3)), "lunes vs mar-jue (sin dispersion)")
het.append(r)
r, _ = efecto_suc(sin_disp & (v.dias_al_fin_mes == 0), sin_disp & (v.dias_al_fin_mes > 0) & ~v.es_quincena.astype(bool), "fin_de_mes vs normal")
het.append(r)
r, _ = efecto_suc(sin_disp & (v.regimen == "IMPAR"), sin_disp & (v.regimen == "PAR"), "impar vs par (sin dispersion)")
het.append(r)
het = pd.DataFrame(het)
het.to_csv(out7 / "heterogeneidad.csv", index=False)
print("\nEV-07 heterogeneidad:\n", het.round(3).to_string())

guardar_json("EV-06", {"R2": m_full.rsquared, "n_fechas": int(m_full.nobs), "efectos": t_full.reset_index().to_dict("records"),
                       "ablacion": abl.to_dict("records")})
guardar_json("EV-07", {"heterogeneidad": het.to_dict("records")})
guardar_json("EV-09", res9)
fe.to_csv(out6 / "efecto_fecha.csv", index=False)
