"""EV-16b: localizar la divergencia. (a) M1 con hash de cada estado interno por iteración; (b) aritmética pura de
NumPy repetida (producto de matrices y sumas) comparada contra la primera iteración.

  python evidencias/ev16b_localizar.py 40
Escribe docs/evidencias/EV-16/localizar.json
"""
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from perfil_retiro.config import cargar_config  # noqa: E402
from perfil_retiro.modelo import M1  # noqa: E402
from comun import calendario_op  # noqa: E402


def hobj(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return hashlib.sha256(pd.util.hash_pandas_object(x.round(9), index=True).values.tobytes()).hexdigest()[:10]
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:10]


N = int(sys.argv[1])
cfg = cargar_config()
cal = calendario_op()
p = pd.read_parquet(RAIZ / "out" / "cache" / "panel_20240101.parquet")
D, T = 20261030, 20261103
v = p[p.valido & (p.id_fecha <= D) & (p.id_fecha >= 20250602)].copy()
v["evento_pe"] = False
n = v.groupby("id_sucursal").size()
v = v[v.id_sucursal.isin(n[n >= 20].index)]
obj = pd.DataFrame({"id_sucursal": n[n >= 20].index, "id_fecha": T})
cT = cal.set_index("id_fecha").loc[T]
for c in cT.index:
    obj[c] = cT[c]
obj["evento_pe"] = False
filas = []
for i in range(N):
    m = M1().ajustar(v)
    pr = m.predecir(obj)
    filas.append({"beta": hobj(m.beta), "dev": hobj(m.dev), "dev_letras": hobj(m.dev_letras), "nivel": hobj(m.nivel),
                  "qs": hobj(m.qs), "qg": hobj(m.qg), "pred": hobj(pr.select_dtypes("number"))})
    print(i, filas[-1], flush=True)
F = pd.DataFrame(filas)
modo = F.mode().iloc[0]
anom = F[(F != modo).any(axis=1)]
primer_paso = [next((c for c in F.columns if r[c] != modo[c]), None) for _, r in anom.iterrows()]
# (b) aritmética pura
rng = np.random.default_rng(0)
A, B = rng.normal(size=(1500, 1500)), rng.normal(size=(1500, 1500))
ref = None
dist = 0
for i in range(N * 3):
    h = hobj((A @ B).sum(axis=0)) + hobj(np.sort(A.ravel())[::997])
    ref = ref or h
    dist += h != ref
res = {"N_m1": N, "anomalias_m1": int(len(anom)), "primer_estado_distinto": primer_paso,
       "estados_por_iteracion": filas, "N_numpy": N * 3, "anomalias_numpy": int(dist)}
d = RAIZ / "docs" / "evidencias" / "EV-16"
(d / "localizar.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("N_m1", "anomalias_m1", "primer_estado_distinto", "N_numpy", "anomalias_numpy")}))
