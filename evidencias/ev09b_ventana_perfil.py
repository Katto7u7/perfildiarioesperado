"""EV-09b ¿Qué ventana de perfil describe mejor el comportamiento del mes siguiente?

Origen: primer día operativo de cada mes 2026-01 .. 2026-11 (datos estrictamente anteriores).
Candidatos (estadísticos del retiro diario por sucursal, solo días OK):
  V120 / V180 / V250 : últimos N días operativos válidos (mezcla regímenes)
  R_MISMO            : últimos ~126 días válidos del MISMO régimen (paridad de mes) que el mes objetivo
  R_MISMO_6M         : días del mismo régimen en los últimos 12 meses
Objetivo: media, p90 y p95 realizados en el mes objetivo.
Métrica: error absoluto mediano en log (|log(pred/real)|) entre sucursales y WAPE de la media. IC por bootstrap de sucursales.
"""
import numpy as np
import pandas as pd

from comun import bootstrap_ci, guardar_json, panel_completo, salida

out = salida("EV-09")
p = panel_completo()
v = p[(p.id_fecha >= 20250602) & p.valido][["id_sucursal", "id_fecha", "retiro", "regimen", "mes", "anio"]].copy()
v["ym"] = v.anio * 100 + v.mes
meses = sorted(v.ym.unique())
origenes = [m for m in meses if 202601 <= m <= 202611]


def stats(g):
    return pd.Series({"media": g.mean(), "p90": g.quantile(0.9), "p95": g.quantile(0.95), "n": len(g)})


filas = []
for ym in origenes:
    f0 = int(v[v.ym == ym].id_fecha.min())
    hist = v[v.id_fecha < f0].sort_values("id_fecha")
    reg_obj = "IMPAR" if (ym % 100) % 2 == 1 else "PAR"
    real = v[v.ym == ym].groupby("id_sucursal").retiro.apply(stats).unstack()
    real = real[real.n >= 10]
    cands = {}
    for N in (120, 180, 250):
        cands[f"V{N}"] = hist.groupby("id_sucursal").tail(N)
    same = hist[hist.regimen == reg_obj]
    cands["R_MISMO_126"] = same.groupby("id_sucursal").tail(126)
    lim = int((pd.Timestamp(str(f0)) - pd.DateOffset(months=12)).strftime("%Y%m%d"))
    cands["R_MISMO_12M"] = same[same.id_fecha >= lim]
    for nom, h in cands.items():
        s = h.groupby("id_sucursal").retiro.apply(stats).unstack()
        s = s[s.n >= 40]
        j = s.join(real, lsuffix="_pred", rsuffix="_real", how="inner")
        for st in ("media", "p90", "p95"):
            a, b = j[f"{st}_pred"], j[f"{st}_real"]
            okm = (a > 0) & (b > 0)
            e = np.abs(np.log(a[okm] / b[okm]))
            filas.append({"origen": ym, "regimen_obj": reg_obj, "ventana": nom, "estadistico": st, "n_suc": int(okm.sum()),
                          "err_log_mediano": float(e.median()),
                          "wape": float((a - b).abs().sum() / b.sum()), "sesgo": float((a - b).sum() / b.sum())})
r = pd.DataFrame(filas)
print("filas con error NaN:", r[r.err_log_mediano.isna()][["origen","ventana","estadistico","n_suc"]].to_dict("records"))
r = r.dropna(subset=["err_log_mediano"])
r.to_csv(out / "ventana_perfil_por_origen.csv", index=False)
res = r.groupby(["estadistico", "ventana"]).agg(err_log_mediano=("err_log_mediano", "mean"), wape=("wape", "mean"),
                                                 sesgo=("sesgo", "mean"), meses_ganados=("err_log_mediano", "size")).reset_index()
# meses ganados: por origen y estadístico, qué ventana tuvo menor error
win = r.loc[r.groupby(["origen", "estadistico"]).err_log_mediano.idxmin()]
res["meses_ganados"] = res.apply(lambda x: int(((win.estadistico == x.estadistico) & (win.ventana == x.ventana)).sum()), axis=1)
# IC bootstrap sobre los 11 orígenes de la diferencia R_MISMO_126 - V180 (err media)
ci = {}
for st in ("media", "p90", "p95"):
    a = r[(r.estadistico == st) & (r.ventana == "R_MISMO_126")].set_index("origen").err_log_mediano
    b = r[(r.estadistico == st) & (r.ventana == "V180")].set_index("origen").err_log_mediano
    d = (a - b).dropna().values
    ci[st] = {"dif_media": float(d.mean()), "ic95": bootstrap_ci(d, np.mean, seed=2), "meses_R_mejor": int((d < 0).sum()), "n": len(d)}
res.to_csv(out / "ventana_perfil_resumen.csv", index=False)
guardar_json("EV-09", {**__import__("json").load(open(out / "resultados.json", encoding="utf-8")),
                       "ventana_perfil": res.to_dict("records"), "R_MISMO_126_vs_V180": ci})
pd.set_option("display.width", 200)
print(res.round(4).to_string())
print(ci)
print(r.pivot_table(index="origen", columns="ventana", values="err_log_mediano", aggfunc="first").loc[:, :].round(3)
      if False else r[r.estadistico == "media"].pivot(index="origen", columns="ventana", values="err_log_mediano").round(3))
