"""EV-16: ¿el cómputo es determinista en esta máquina? Ajusta M1 + predice N veces sobre los MISMOS datos (en un
solo proceso) y cuenta resultados distintos (hash de las predicciones) y fallas.

  python evidencias/ev16_estres_determinismo.py 12 actual
  set MKL_CBWR=COMPATIBLE & set MKL_NUM_THREADS=1 & python evidencias/ev16_estres_determinismo.py 12 mkl_fijo
Escribe docs/evidencias/EV-16/estres_<tag>.json
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import pandas as pd  # noqa: E402

from perfil_retiro.config import cargar_config  # noqa: E402
from perfil_retiro.modelo import M1  # noqa: E402
from perfil_retiro.panel import calendario_operativo  # noqa: E402
from perfil_retiro.sources import extraer  # noqa: E402

N, tag = int(sys.argv[1]), sys.argv[2]
cfg = cargar_config()
cal = calendario_operativo(extraer(cfg, "fecha", usar_cache=False), extraer(cfg, "dispersion", usar_cache=False))
p = pd.read_parquet(RAIZ / "out" / "cache" / "panel_20240101.parquet")
D, T = 20261030, 20261103
v = p[p.valido & (p.id_fecha <= D) & (p.id_fecha >= 20250602)].copy()
v["evento_pe"] = False
n = v.groupby("id_sucursal").size()
sucs = n[n >= 20].index
v = v[v.id_sucursal.isin(sucs)]
obj = pd.DataFrame({"id_sucursal": sucs, "id_fecha": T})
cT = cal.set_index("id_fecha").loc[T]
for c in cT.index:
    obj[c] = cT[c]
obj["evento_pe"] = False
hashes, fallas, tiempos = [], [], []
for i in range(N):
    t0 = time.time()
    try:
        pr = M1().ajustar(v).predecir(obj)
        h = hashlib.sha256(pd.util.hash_pandas_object(pr.round(6), index=False).values.tobytes()).hexdigest()[:12]
        hashes.append(h)
    except Exception as e:  # noqa: BLE001
        fallas.append(f"{type(e).__name__}: {str(e)[:120]}")
        hashes.append("FALLA")
    tiempos.append(round(time.time() - t0, 1))
    print(i, hashes[-1], tiempos[-1], flush=True)
res = {"tag": tag, "N": N, "MKL_CBWR": os.environ.get("MKL_CBWR"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
       "resultados_distintos": sorted(set(h for h in hashes if h != "FALLA")), "conteo": pd.Series(hashes).value_counts().to_dict(),
       "fallas": fallas, "segundos": tiempos}
d = RAIZ / "docs" / "evidencias" / "EV-16"
d.mkdir(parents=True, exist_ok=True)
(d / f"estres_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("tag", "conteo", "fallas")}, ensure_ascii=False))
