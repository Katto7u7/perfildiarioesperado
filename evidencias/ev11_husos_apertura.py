"""EV-11 Husos horarios, hora de apertura local y desfase efectivo de los cortes.

Dato del negocio (2026-11-18): las sucursales abren a las 9:00 HORA LOCAL; las muy transaccionales a las 8:00.
Hipótesis a probar:
  H1  los cortes 900/1100/1300/1700 están en hora del CENTRO (no local).
  H2  la apertura es a la misma hora LOCAL en todas las zonas -> al alinear por "horas desde la apertura local"
      las curvas de las distintas zonas coinciden.
  H3  existe un desfase efectivo δ (corte tomado después de la etiqueta, o apertura anterior) estimable.
  H4  las sucursales "muy transaccionales" (apertura 8:00) son identificables: ~1 h más de actividad al primer corte.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from comun import fuente, guardar_json, panel_completo, salida

out = salida("EV-11")
p = panel_completo()
s = fuente("sucursal").set_index("id_sucursal")
OFFSET = {"QUINTANA ROO": 1, "BAJA CALIFORNIA SUR": -1, "SONORA": -1, "SINALOA": -1, "NAYARIT": -1, "BAJA CALIFORNIA": -2}
f = p[(p.id_fecha >= 20250602) & p.forma_valida].copy()
f["entidad"] = f.id_sucursal.map(s.entidad)
f["offset"] = f.entidad.map(OFFSET).fillna(0).astype(int)
fecha = pd.to_datetime(f.id_fecha.astype(str))
dst = ((fecha.dt.month > 3) & (fecha.dt.month < 11)) | ((fecha.dt.month == 3) & (fecha.dt.day >= 8)) | ((fecha.dt.month == 11) & (fecha.dt.day <= 7))
f.loc[(f.entidad == "BAJA CALIFORNIA") & dst, "offset"] = -1          # BC sigue el horario de verano de EUA
f["zona"] = f.offset.map({1: "SURESTE(+1)", 0: "CENTRO", -1: "PACIFICO(-1)", -2: "NOROESTE_INV(-2)"})

# H1/H2: tabla de alineación. tau = hora de corte (centro) - apertura local expresada en hora centro (9 - offset)
rows = []
for c in (900, 1100, 1300):
    g = f.groupby("zona").agg(p=(f"p_{c}", "median"), off=("offset", "first"), n_suc=("id_sucursal", "nunique"))
    for z, r in g.iterrows():
        rows.append({"zona": z, "corte_centro": c, "tau_horas_desde_9_local": c / 100 - (9 - r.off), "fraccion_mediana": r.p, "n_suc": r.n_suc})
al = pd.DataFrame(rows).sort_values(["tau_horas_desde_9_local", "zona"])
al.to_csv(out / "alineacion_zonas.csv", index=False)

# H3: ajuste común cóncavo F(tau) = 1 - exp(-k (tau+δ)) (b=1: saturante; la forma convexa b libre se fue al límite) sobre los puntos zona-corte con tau <= 3 (antes de que influya el cierre)
pts = al[(al.tau_horas_desde_9_local <= 3) & (al.n_suc >= 20)]
w = np.sqrt(pts.n_suc.values)


def resid(th):
    k, b, d = th
    t = np.clip(pts.tau_horas_desde_9_local.values + d, 1e-6, None)
    return w * ((1 - np.exp(-k * t ** b)) - pts.fraccion_mediana.values)


fit = least_squares(resid, x0=[0.3, 1.0, 0.5], bounds=([1e-3, 0.999, -0.5], [5, 1.001, 2.5]))
k, b, delta = fit.x
# IC de δ por bootstrap de sucursales
rng = np.random.default_rng(7)
suc = f.id_sucursal.unique()
boots = []
med_suc = f.groupby(["id_sucursal", "zona", "offset"])[["p_900", "p_1100", "p_1300"]].median().reset_index()
for _ in range(200):
    sm = med_suc.iloc[rng.integers(0, len(med_suc), len(med_suc))]
    rr = []
    for c in (900, 1100, 1300):
        g = sm.groupby(["zona", "offset"])[f"p_{c}"].agg(["median", "size"]).reset_index()
        for _, r in g.iterrows():
            tau = c / 100 - (9 - r.offset)
            if tau <= 3 and r["size"] >= 20:
                rr.append((tau, r["median"], r["size"]))
    rr = np.array(rr)

    def rb(th):
        t = np.clip(rr[:, 0] + th[2], 1e-6, None)
        return np.sqrt(rr[:, 2]) * ((1 - np.exp(-th[0] * t ** th[1])) - rr[:, 1])
    boots.append(least_squares(rb, x0=fit.x, bounds=([1e-3, 0.999, -0.5], [5, 1.001, 2.5])).x[2])
ci = (float(np.quantile(boots, .025)), float(np.quantile(boots, .975)))

# H4: sucursales con ~1 h extra de actividad al primer corte de su zona -> apertura 8:00 local
F = lambda t: 1 - np.exp(-k * np.clip(t + delta, 1e-6, None) ** b)
ms = f.groupby(["id_sucursal", "offset"])[["p_900", "p_1100"]].median().reset_index()
ms = ms[ms.offset == 0]
esp9, esp8 = F(0.0), F(1.0)                  # fracción esperada al corte 9 (centro) si abre 9:00 vs 8:00
umbral = (esp9 + esp8) / 2
ms["apertura_inferida"] = np.where(ms.p_900 > umbral, 800, 900)
ms.to_csv(out / "apertura_inferida_centro.csv", index=False)
g8 = ms[ms.apertura_inferida == 800]
res = {
    "H1_H2_alineacion": al.to_dict("records"),
    "ajuste": {"k": k, "b": b, "delta_horas": delta, "delta_minutos": delta * 60, "ic95_delta_horas": ci, "rmse": float(np.sqrt(np.mean((fit.fun / w) ** 2)))},
    "fraccion_esperada_primer_corte": {"abre_9_local": float(esp9), "abre_8_local": float(esp8), "umbral": float(umbral)},
    "H4_centro_inferidas_8am": {"n": int(len(g8)), "de": int(len(ms)), "p1100_mediana_8am": float(g8.p_1100.median()),
                                "p1100_mediana_9am": float(ms[ms.apertura_inferida == 900].p_1100.median()),
                                "p1100_esperado_si_8am": float(F(2.0 + 1.0)), "p1100_esperado_si_9am": float(F(2.0))},
}
guardar_json("EV-11", res)
pd.set_option("display.width", 200)
print(al.round(3).to_string())
print(res["ajuste"], res["fraccion_esperada_primer_corte"], res["H4_centro_inferidas_8am"], sep="\n")
