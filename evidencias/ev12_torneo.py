"""EV-12 Torneo de modelos sobre el periodo de DESARROLLO (o el holdout, una sola vez). Protocolo: docs/03.

Uso: python ev12_torneo.py ETIQUETA_BT [EV-12|EV-13]
"""
import json
import sys

import numpy as np
import pandas as pd

from comun import RAIZ, guardar_json, salida
from perfil_retiro.calibracion import QCOLS, conformal_causal, riesgo
from perfil_retiro.evaluacion import bootstrap_bloques, calibracion_prob, dm_test, pinball, por_sucursal, sesgo, wape
from perfil_retiro.modelo import QUANTS

tag = sys.argv[1]
EV = sys.argv[2] if len(sys.argv) > 2 else "EV-12"
EVAL_DESDE = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # holdout: solo se evalúan objetivos >= esta fecha
GANADOR_FIJO = sys.argv[4] if len(sys.argv) > 4 else None           # holdout: el ganador NO se vuelve a elegir
out = salida(EV)
bt = pd.read_parquet(RAIZ / "out" / "backtest" / f"bt_{tag}.parquet")
am = bt.estrato == "AM"
bt["m1"] = bt.p50
bt["ens_am"] = np.where(am, (bt.p50 + bt.nb) / 2, bt.p50)
if "m2" in bt:
    bt["ens_m1m2"] = (bt.p50 + bt.m2) / 2
    bt["ens_all"] = np.where(am, (bt.p50 + bt.nb + bt.m2) / 3, (bt.p50 + bt.m2) / 2)
cands = [c for c in ["b0", "b1", "b2", "nb", "m1", "m2", "ens_am", "ens_m1m2", "ens_all"] if c in bt]
ev = bt[(bt.estado_real == "OK") & bt.real.notna() & (bt.id_fecha >= EVAL_DESDE)].dropna(subset=cands).copy()
res = {"etiqueta": tag, "filas": int(len(ev)), "origenes": int(ev.origen.nunique()), "sucursales": int(ev.id_sucursal.nunique()),
       "rango_objetivo": [int(ev.id_fecha.min()), int(ev.id_fecha.max())]}

# ---------------------------------------------------------------- 1. puntuales
tab = []
for c in cands:
    e = ev[c] - ev.real
    tab.append({"modelo": c, "WAPE": wape(ev[c], ev.real), "sesgo_punto": sesgo(ev[c], ev.real),
                "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                "err_log_mediano": float(np.abs(np.log1p(ev[c]) - np.log1p(ev.real)).median())})
tab = pd.DataFrame(tab).set_index("modelo")
fx = ev.dropna(subset=["fenix"]) if "fenix" in ev else ev.iloc[:0]
if len(fx):
    tab.loc["fenix (subconjunto)", ["WAPE", "sesgo_punto"]] = [wape(fx.fenix, fx.real), sesgo(fx.fenix, fx.real)]
    for c in cands:
        tab.loc[c, "WAPE_subconjunto_fenix"] = wape(fx[c], fx.real)
tab.to_csv(out / "metricas_puntuales.csv")

# por segmento
def seg(df, col):
    return df.groupby(col).apply(lambda t: pd.Series({c: wape(t[c], t.real) for c in cands} | {"n": len(t)}), include_groups=False)


ev["mes"] = ev.id_fecha // 100
ev["tam"] = pd.qcut(ev.groupby("id_sucursal").real.transform("mean").rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
segs = {k: seg(ev, k) for k in ["estrato", "mes", "tam"]}
for k, t in segs.items():
    t.to_csv(out / f"wape_por_{k}.csv")

# ---------------------------------------------------------------- 2. significancia vs B1
d_loss = {c: ev.groupby("id_fecha").apply(lambda t: np.abs(t[c] - t.real).sum() / t.real.sum(), include_groups=False) for c in cands}
sig = []
for c in cands:
    if c == "b1":
        continue
    st, pv = dm_test(d_loss[c], d_loss["b1"])
    lo, hi = bootstrap_bloques(ev, c, "b1", n=500, seed=3)
    sig.append({"modelo": c, "WAPE": tab.loc[c, "WAPE"], "WAPE_b1": tab.loc["b1", "WAPE"],
                "mejora_rel_vs_b1": 1 - tab.loc[c, "WAPE"] / tab.loc["b1", "WAPE"], "DM_stat": st, "DM_p": pv,
                "IC95_dif_WAPE": [lo, hi]})
sig = pd.DataFrame(sig).set_index("modelo")
sig.to_csv(out / "significancia_vs_b1.csv")
ps = por_sucursal(ev, cands)
ps.to_csv(out / "wape_por_sucursal.csv")
no_peor = {c: float((ps[c] <= ps.b1 + 1e-12).mean()) for c in cands}

# ---------------------------------------------------------------- 3. ganador y comparación entre finalistas
best = GANADOR_FIJO or tab.loc[[c for c in cands if c not in ("b0", "b1", "b2", "nb")], "WAPE"].idxmin()
fin = {}
for c in cands:
    if c in (best, "b0", "b2"):
        continue
    st, pv = dm_test(d_loss[best], d_loss[c])
    fin[c] = {"DM_stat": st, "DM_p": pv, "mejora_rel": 1 - tab.loc[best, "WAPE"] / tab.loc[c, "WAPE"]}

# ---------------------------------------------------------------- 4. probabilístico (conformal causal del ganador y de M1)
prob = {}
cal_bt = {}
for c in dict.fromkeys([best, "m1"]):
    b2 = conformal_causal(bt, c)
    b2 = riesgo(b2, factor=1.0)
    cal_bt[c] = b2
    e = b2[(b2.estado_real == "OK") & b2.real.notna() & (b2.c_fuente == "CONFORMAL") & (b2.id_fecha >= EVAL_DESDE)]
    inside = (e.real >= e.c_q10) & (e.real <= e.c_q90)
    cov_suc = inside.groupby(e.id_sucursal).mean()
    width = (e.c_q90 - e.c_q10) / e[c].replace(0, np.nan)
    cond = inside.groupby(pd.qcut(width.rank(method="first"), 10, labels=False)).mean()
    pit = np.zeros(len(e))
    qm = e[["c_" + q for q in QCOLS]].values
    for i in range(len(e)):
        pit[i] = np.interp(e.real.values[i], qm[i], QUANTS, left=0.0, right=1.0)
    prob[c] = {
        "n": int(len(e)), "cobertura_P10_P90": float(inside.mean()), "cobertura_P95": float((e.real <= e.c_q95).mean()),
        "cobertura_P05_P95": float(((e.real >= e.c_q05) & (e.real <= e.c_q95)).mean()),
        "pct_suc_cobertura_70_90": float(((cov_suc >= 0.70) & (cov_suc <= 0.90)).mean()),
        "cobertura_por_decil_ancho": cond.round(3).tolist(),
        "cobertura_por_estrato": inside.groupby(e.estrato).mean().round(3).to_dict(),
        "cobertura_por_mes": inside.groupby(e.id_fecha // 100).mean().round(3).to_dict(),
        "pinball": {q: pinball(e["c_" + q], e.real, a) for q, a in zip(QCOLS, QUANTS)},
        "PIT_hist_10": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
        "sesgo_media_conformal": sesgo(e.c_media, e.real),
        "sesgo_media_por_estrato": e.groupby("estrato").apply(lambda t: sesgo(t.c_media, t.real), include_groups=False).round(3).to_dict(),
    }
    # riesgo
    rr = b2[(b2.estado_real == "OK") & b2.r_evento_real.notna() & b2.r_p_critico.notna() & (b2.c_fuente == "CONFORMAL")
            & (b2.id_fecha >= EVAL_DESDE)]
    limpio = rr[rr.forma_valida_real.astype(bool)]
    for nom, dfr in (("todos", rr), ("sin_traspaso", limpio)):
        tcal, met = calibracion_prob(dfr.r_p_critico.values, dfr.r_evento_real.values)
        tcal.to_csv(out / f"calibracion_riesgo_{c}_{nom}.csv")
        alert = dfr.r_p_critico >= 0.5
        tp = (alert & (dfr.r_evento_real == 1)).sum()
        met.update({"precision@0.5": float(tp / max(alert.sum(), 1)), "recall@0.5": float(tp / max((dfr.r_evento_real == 1).sum(), 1)),
                    "max_error_decil_pp": float(tcal.error_abs_pp[tcal.n >= 200].max())})
        prob[c][f"riesgo_{nom}"] = met

# ---------------------------------------------------------------- 5. criterios del protocolo (§6)
w = best
pc = prob[w]
crit = {
    "1_mejora_10pct_vs_b1_significativa": bool(sig.loc[w, "mejora_rel_vs_b1"] >= 0.10 and sig.loc[w, "DM_p"] < 0.05 and sig.loc[w, "IC95_dif_WAPE"][1] < 0),
    "2_no_peor_que_b1_en_70pct_sucursales": bool(no_peor[w] >= 0.70),
    "3_sesgo_media_<=3pct": bool(abs(pc["sesgo_media_conformal"]) <= 0.03),
    "3b_sesgo_por_estrato_<=5pct": bool(all(abs(v) <= 0.05 for v in pc["sesgo_media_por_estrato"].values())),
    "4_cobertura_global_77_83": bool(0.77 <= pc["cobertura_P10_P90"] <= 0.83),
    "4b_85pct_suc_cobertura_70_90": bool(pc["pct_suc_cobertura_70_90"] >= 0.85),
    "5_riesgo_calibrado_<=5pp": bool(pc["riesgo_sin_traspaso"]["max_error_decil_pp"] <= 5),
}
res.update({"metricas": tab.round(4).reset_index().to_dict("records"), "significancia": sig.reset_index().to_dict("records"),
            "no_peor_que_b1_pct_suc": no_peor, "ganador": w, "ganador_vs_otros_DM": fin, "probabilistico": prob,
            "criterios": crit})
guardar_json(EV, res)
for c, b2 in cal_bt.items():
    b2.to_parquet(RAIZ / "out" / "backtest" / f"cal_{tag}_{c}.parquet", index=False)

pd.set_option("display.width", 220)
print(tab.round(4).to_string())
print("\nSignificancia vs B1:\n", sig.round(4).to_string())
print("\nNo peor que B1 (% sucursales):", {k: round(v, 3) for k, v in no_peor.items()})
print("\nGanador:", w, "\nvs otros:", json.dumps(fin, indent=0, default=float)[:800])
print("\nWAPE por estrato:\n", segs["estrato"].round(3).to_string())
print("\nWAPE por mes:\n", segs["mes"].round(3).to_string())
for c in prob:
    pp = {k: v for k, v in prob[c].items() if k not in ("pinball", "cobertura_por_mes")}
    print(f"\nProbabilístico {c}:", json.dumps(pp, indent=0, default=float)[:1500])
print("\nCRITERIOS:", json.dumps(crit, indent=1))
