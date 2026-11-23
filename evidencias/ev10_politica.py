"""EV-10 Concordancia con la política preliminar y efecto del error de fines de semana.

1) Impacto por sucursal del error (fines de semana copia del viernes sumados al lunes): motor original vs copia corregida
   (ambas corridas en out/ev10, sin tocar Politica_sucursal/).
2) Concordancia del umbral crítico relativo (P90 retiro diario, mismo régimen, MOVIMIENTOS) con saldo_minimo_operativo
   y stock_seguridad_base de la política.
3) SOBREDOTADA: % de días con cierre > saldo_maximo_operativo / > capacidad.
"""
import numpy as np
import pandas as pd
from scipy import stats

from comun import RAIZ, guardar_json, panel_completo, salida
from perfil_retiro.sources import leer_tope
from comun import CFG

out = salida("EV-10")
d = RAIZ / "out" / "ev10"
res = {}
cols = ["stock_seguridad_base", "saldo_minimo_operativo", "saldo_maximo_operativo"]
imp = {}
for reg in ("IMPAR", "PAR"):
    o = pd.read_csv(d / "orig" / f"politica_{reg}.csv").set_index("id_sucursal")
    f = pd.read_csv(d / "fix" / f"politica_{reg}.csv").set_index("id_sucursal")
    o, f = o[o.in_service == 1], f.loc[o[o.in_service == 1].index]
    r = {}
    for c in cols:
        dif = o[c] - f[c]
        r[c] = {"suma_orig_M": o[c].sum() / 1e6, "suma_fix_M": f[c].sum() / 1e6, "exceso_M": dif.sum() / 1e6,
                "pct_sucursales_afectadas": float((dif.abs() > 0).mean()),
                "exceso_p90_sucursal": float(dif[dif > 0].quantile(.9)) if (dif > 0).any() else 0.0,
                "exceso_max_sucursal": float(dif.max())}
    imp[reg] = r
    top = (o.saldo_minimo_operativo - f.saldo_minimo_operativo).sort_values(ascending=False).head(15)
    top.rename("exceso_saldo_minimo").to_csv(out / f"top_afectadas_{reg}.csv")
res["impacto_error_fin_de_semana"] = imp

# 2) concordancia umbral relativo vs política vigente (régimen del mes vigente = IMPAR, nov-2026)
p = panel_completo()
v = p[(p.id_fecha >= 20250602) & p.valido & (p.regimen == "IMPAR")]
v = v.sort_values("id_fecha").groupby("id_sucursal").tail(126)
p90 = v.groupby("id_sucursal").retiro.quantile(0.9).rename("p90_mismo_regimen")
med = v.groupby("id_sucursal").retiro.median().rename("mediana")
pol = pd.read_csv(d / "orig" / "politica_IMPAR.csv").set_index("id_sucursal")
pol = pol[pol.in_service == 1].join(p90).join(med).dropna(subset=["p90_mismo_regimen"])
pol = pol[pol.p90_mismo_regimen > 0]
res["concordancia_umbral"] = {
    "n": int(len(pol)),
    "spearman_p90_vs_saldo_minimo": float(stats.spearmanr(pol.p90_mismo_regimen, pol.saldo_minimo_operativo).statistic),
    "spearman_p90_vs_stock_seguridad": float(stats.spearmanr(pol.p90_mismo_regimen, pol.stock_seguridad_base).statistic),
    "razon_saldo_minimo_sobre_p90_mediana": float((pol.saldo_minimo_operativo / pol.p90_mismo_regimen).median()),
    "razon_saldo_minimo_sobre_p90_p10_p90": [float((pol.saldo_minimo_operativo / pol.p90_mismo_regimen).quantile(q)) for q in (.1, .9)],
    "pct_saldo_minimo_menor_que_p90": float((pol.saldo_minimo_operativo < pol.p90_mismo_regimen).mean()),
}

# 3) cierres por encima del máximo de política / capacidad (últimos 126 días válidos de cualquier régimen)
s = p[(p.id_fecha >= 20260501) & p.saldo_valido][["id_sucursal", "id_fecha", "saldo_cierre", "regimen"]]
polI = pd.read_csv(d / "orig" / "politica_IMPAR.csv").set_index("id_sucursal")
polP = pd.read_csv(d / "orig" / "politica_PAR.csv").set_index("id_sucursal")
s["smax"] = np.where(s.regimen == "IMPAR", s.id_sucursal.map(polI.saldo_maximo_operativo), s.id_sucursal.map(polP.saldo_maximo_operativo))
tope = leer_tope(CFG).set_index("id_sucursal").capacidad_boveda
s["cap"] = s.id_sucursal.map(tope)
g = s.dropna(subset=["smax"]).groupby("id_sucursal").agg(pct_sobre_max=("saldo_cierre", "size"))
s2 = s.dropna(subset=["smax"])
g = pd.DataFrame({"pct_sobre_max": (s2.saldo_cierre > s2.smax).groupby(s2.id_sucursal).mean(),
                  "pct_sobre_cap": (s2.saldo_cierre > s2.cap).groupby(s2.id_sucursal).mean()})
res["sobredotacion"] = {
    "n_suc": int(len(g)),
    "pct_dias_sobre_max_global": float((s2.saldo_cierre > s2.smax).mean()),
    "pct_dias_sobre_capacidad_global": float((s2.saldo_cierre > s2.cap).mean()),
    "distribucion_pct_dias_sobre_max_por_sucursal": g.pct_sobre_max.quantile([.25, .5, .75, .9]).round(3).to_dict(),
    "sucursales_con_mas_de_50pct_dias_sobre_max": int((g.pct_sobre_max > 0.5).sum()),
}
g.to_csv(out / "sobredotacion_por_sucursal.csv")
guardar_json("EV-10", res)
import json
print(json.dumps(res, indent=1, default=float)[:4000])
