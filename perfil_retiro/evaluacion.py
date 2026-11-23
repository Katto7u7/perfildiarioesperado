"""Métricas y pruebas del protocolo (docs/03 §5-§6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wape(p, y):
    return float(np.abs(p - y).sum() / y.sum())


def sesgo(p, y):
    return float((p - y).sum() / y.sum())


def pinball(q, y, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def dm_test(la: pd.Series, lb: pd.Series):
    """Diebold-Mariano sobre series diarias de pérdida (mismo índice). H0: igual precisión. HAC Newey-West."""
    d = (la - lb).dropna().values
    T = len(d)
    L = int(np.ceil(T ** (1 / 3)))
    dc = d - d.mean()
    v = np.dot(dc, dc) / T
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * np.dot(dc[k:], dc[:-k]) / T
    stat = d.mean() / np.sqrt(v / T)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def bootstrap_bloques(df, a, b, n=1000, seed=0):
    """IC95 de WAPE(a) - WAPE(b) re-muestreando bloques sucursal x semana."""
    x = df.assign(sem=pd.to_datetime(df.id_fecha.astype(str)).dt.isocalendar().week.astype(int) +
                  100 * (df.id_fecha // 10000))
    g = x.groupby(["id_sucursal", "sem"]).apply(
        lambda t: pd.Series({"ea": np.abs(t[a] - t.real).sum(), "eb": np.abs(t[b] - t.real).sum(), "y": t.real.sum()}),
        include_groups=False)
    arr = g.values
    rng = np.random.default_rng(seed)
    res = np.empty(n)
    for i in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        res[i] = (s[0] - s[1]) / s[2]
    return float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def por_sucursal(df, modelos):
    g = df.groupby("id_sucursal")
    out = pd.DataFrame({m: g.apply(lambda t: np.abs(t[m] - t.real).sum() / max(t.real.sum(), 1), include_groups=False)
                        for m in modelos})
    out["n"] = g.size()
    out["retiro_medio"] = g.real.mean()
    return out


def calibracion_prob(p, y, bins=10):
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["dec"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    t = d.groupby("dec").agg(p_media=("p", "mean"), tasa_real=("y", "mean"), n=("y", "size"))
    t["error_abs_pp"] = (t.p_media - t.tasa_real).abs() * 100
    brier = float(np.mean((d.p - d.y) ** 2))
    base = d.y.mean()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(d.y, d.p)) if d.y.nunique() > 1 else np.nan
    return t, {"brier": brier, "brier_climatologia": float(base * (1 - base)), "auc": auc, "tasa_base": float(base),
               "n": int(len(d))}
