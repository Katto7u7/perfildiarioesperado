"""Utilidades compartidas por los scripts de evidencia."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from perfil_retiro.config import cargar_config  # noqa: E402
from perfil_retiro.sources import extraer  # noqa: E402

CFG = cargar_config()
DOCS = RAIZ / "docs" / "evidencias"
HASTA = 29991231


def fuente(nombre):
    return extraer(CFG, nombre, 20240101, HASTA)


def to_date(s):
    return pd.to_datetime(s.astype(int).astype(str), format="%Y%m%d")


def salida(ev: str) -> Path:
    d = DOCS / ev
    d.mkdir(parents=True, exist_ok=True)
    return d


def guardar_json(ev: str, obj: dict):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        raise TypeError(type(o))
    (salida(ev) / "resultados.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def bootstrap_ci(x, stat=np.mean, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    b = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def calendario():
    f = fuente("fecha")
    f["fecha"] = to_date(f.id_fecha)
    f["dow"] = f.fecha.dt.dayofweek          # 0 = lunes
    f["habil_cal"] = (f.dow < 5) & (f.is_holiday == 0)
    return f


def panel_completo(desde=20240101, usar_cache=True):
    """Panel limpio 2024-01 -> última fecha, con detectores automáticos de calidad (EV-01)."""
    from perfil_retiro.calidad import detectar_copias, detectar_parciales, mov_diario
    from perfil_retiro.panel import calendario_operativo, construir_panel
    cache = RAIZ / "out" / "cache" / f"panel_{desde}.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)
    mv, mon = fuente("movimientos_diario"), fuente("monitoreo_cortes")
    cal = calendario_operativo(fuente("fecha"), fuente("dispersion"))
    dia = mov_diario(mv)
    cop = detectar_copias(dia)
    par = detectar_parciales(dia, mon[mon.hora_corte == 1700])
    fc = int(mv.id_fecha.max())
    p = construir_panel(mv, mon, cal, CFG, fc, copias=set(cop.id_fecha), parciales=set(par[par.parcial].id_fecha),
                        sospechosos=set(par[par.sospechoso].id_fecha), desde=desde)
    p.to_parquet(cache, index=False)
    return p


def calendario_op():
    from perfil_retiro.panel import calendario_operativo
    return calendario_operativo(fuente("fecha"), fuente("dispersion"))
