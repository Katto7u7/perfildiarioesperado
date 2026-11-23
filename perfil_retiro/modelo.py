"""Pronóstico del retiro del siguiente día operativo.

M1 (docs/02 §1): log1p(retiro) = nivel(s) + Σ efectos comunes(fecha) + Σ efectos propios contraídos(s) + ε
  - efectos comunes estimados a nivel fecha (EV-06): día semana, inicio de mes, quincena, monto AM, BECAS día 2, evento LEEN
  - efectos propios contraídos (EV-07): AM día 1, AM día 2, BECAS día 1, régimen IMPAR
  - nivel: media exponencial causal de residuos (captura tendencia/cambios de nivel)
  - cuantiles: residuos un-paso-adelante (causales) por estrato (día AM / no AM), sucursal mezclada con global

Todas las funciones reciben SOLO historia <= fecha de corte. Sin estado oculto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
PROPIOS = ["am1", "am2", "becas1", "impar"]
COMUNES = ["martes", "miercoles", "jueves", "viernes", "inicio_mes", "quincena", "log_monto_am", "becas2", "evento_pe",
           "stock_am_h2", "stock_am_h8", "stock_becas_h2", "stock_becas_h8"]
POS = [f"{n}_{r}" for n in ("pos01_03", "pos04_08", "pos15_19", "pos20_") for r in ("I", "P")]


# --------------------------------------------------------------------------- variables
def variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario a partir de columnas del calendario operativo (+ evento_pe si existe)."""
    X = pd.DataFrame(index=df.index)
    am, b = df.disp_am.fillna(False).astype(bool), df.disp_becas.fillna(False).astype(bool)
    X["am1"] = (am & df.disp_am_primero.fillna(False).astype(bool)).astype(float)
    X["am2"] = (am & df.disp_am_segundo.fillna(False).astype(bool) & ~df.disp_am_primero.fillna(False).astype(bool)).astype(float)
    X["becas1"] = (b & df.disp_becas_primero.fillna(False).astype(bool)).astype(float)
    X["becas2"] = (b & df.disp_becas_segundo.fillna(False).astype(bool) & ~df.disp_becas_primero.fillna(False).astype(bool)).astype(float)
    X["impar"] = (df.regimen == "IMPAR").astype(float)
    for d, n in ((1, "martes"), (2, "miercoles"), (3, "jueves"), (4, "viernes")):
        X[n] = (df.dow == d).astype(float)
    X["inicio_mes"] = (df.dia_habil_mes <= 3).astype(float)
    # posición en el mes por régimen (referencia: días hábiles 9-14). Captura arrastre de AM al inicio del mes par,
    # rampa de BECAS (calendario desfasado del retiro real) y progresión de letras.
    dh = df.dia_habil_mes
    reg = np.where(df.regimen == "IMPAR", "I", "P")
    for nom, lo, hi in (("pos01_03", 1, 3), ("pos04_08", 4, 8), ("pos15_19", 15, 19), ("pos20_", 20, 99)):
        for r in ("I", "P"):
            X[f"{nom}_{r}"] = ((dh >= lo) & (dh <= hi) & (reg == r)).astype(float)
    X["quincena"] = df.es_quincena.fillna(False).astype(float)
    m = np.log(df.disp_am_monto.where(df.disp_am_monto > 0))
    X["log_monto_am"] = (m - 22.5).fillna(0.0) * X.am1.add(X.am2).clip(upper=1)   # centrado aprox. (e^22.5 ≈ 5.9e9)
    for c in ("stock_am_h2", "stock_am_h8", "stock_becas_h2", "stock_becas_h8"):
        X[c] = df[c].astype(float).fillna(0.0) if c in df else 0.0
    X["evento_pe"] = df["evento_pe"].astype(float) if "evento_pe" in df else 0.0
    X["estrato"] = np.where((X.am1 + X.am2) > 0, "AM", "NO_AM")
    return X


def contexto_b1(X: pd.DataFrame) -> pd.Series:
    t = np.select([X.am1 > 0, X.am2 > 0, X.becas1 > 0, X.becas2 > 0], ["AM1", "AM2", "BE1", "BE2"], default="SIN")
    return pd.Series(t, index=X.index) + "_" + np.where(X.impar > 0, "I", "P")


# --------------------------------------------------------------------------- M1
class M1:
    """regimen=True (por defecto, derivado de EV-09 y del diagnóstico del 2026-11-18): nivel propio por
    sucursal x régimen y posición-en-mes por régimen; el efecto IMPAR queda absorbido en los niveles.
    regimen=False: versión original (nivel único + dummy IMPAR contraída), conservada para el torneo."""

    def __init__(self, vida_media: float = 4.0, n0_cuantil: float = 40.0, max_dias: int = 400, regimen: bool = True,
                 letras: bool = True, k_letras: float = 2.0, ridge: float = 0.01):
        self.ridge = ridge
        self.letras = letras          # desviación propia por combinación de letras (columna 'llave_letras')
        self.k_letras = k_letras      # contracción: w = n / (n + k)
        self.h = vida_media
        self.n0 = n0_cuantil
        self.max_dias = max_dias
        self.regimen = regimen
        self.propios = ["am1", "am2", "becas1"] if regimen else PROPIOS
        comunes = [c for c in COMUNES if not (regimen and c == "inicio_mes")]
        self.cols_fecha = [c for c in self.propios + comunes + (POS if regimen else []) if c != "evento_pe"]

    def _llave(self, df):
        if self.regimen:
            return df.id_sucursal.astype(str) + "|" + df.regimen.astype(str)
        return df.id_sucursal.astype(str)

    def ajustar(self, hist: pd.DataFrame):
        """hist: filas válidas (id_sucursal, id_fecha, retiro, + calendario), solo <= corte."""
        h = hist.sort_values(["id_sucursal", "id_fecha"]).groupby("id_sucursal").tail(self.max_dias).copy()
        X = variables(h)
        h["y"] = np.log1p(h.retiro.values)
        h["k"] = self._llave(h).values
        # 1) efectos globales a nivel fecha (efecto fijo de sucursal[ x régimen])
        h["y_dm"] = h.y - h.groupby("k").y.transform("mean")
        fe = h.groupby("id_fecha").y_dm.mean()
        Xd = X.groupby(h.id_fecha)[self.cols_fecha].first().loc[fe.index]
        Xd = Xd.loc[:, Xd.std() > 0]
        # Mínimos cuadrados acotados + ridge ligero: los saldos pendientes de dispersión solo pueden AUMENTAR el
        # retiro (β >= 0, restricción física); sin ella, la confusión con el nivel produce signos imposibles.
        from scipy.optimize import lsq_linear
        A = np.column_stack([np.ones(len(Xd)), Xd.values])
        lo = np.full(A.shape[1], -np.inf)
        hi = np.full(A.shape[1], np.inf)
        for j, c in enumerate(Xd.columns, start=1):
            if c.startswith("stock_") or c in ("am1", "am2", "becas1", "log_monto_am"):
                lo[j] = 0.0
        lam = self.ridge * len(Xd)
        R = np.sqrt(lam) * np.eye(A.shape[1])
        R[0, 0] = 0.0
        sol = lsq_linear(np.vstack([A, R]), np.r_[fe.values, np.zeros(A.shape[1])], bounds=(lo, hi))
        self.beta = pd.Series(sol.x[1:], index=Xd.columns)
        bcols = list(self.beta.index)
        base = h.y - X[bcols].values @ self.beta.values
        h["r0"] = base - base.groupby(h.k).transform("mean")
        if X.evento_pe.sum() >= 200:
            ev = X.evento_pe > 0
            self.beta["evento_pe"] = float(h.r0[ev].mean() - h.r0[~ev].mean())
        else:
            self.beta["evento_pe"] = 0.0
        bcols = list(self.beta.index)
        # 2) desviaciones propias contraídas (empírico-bayes)
        h["r"] = h.y - X[bcols].values @ self.beta.values
        h["r"] = h.r - h.groupby("k").r.transform("mean")
        dev, tau2 = {}, {}
        ref_all = X[["am1", "am2", "becas1"]].sum(axis=1) == 0
        for k in self.propios:
            if k == "impar":
                on, off = (X.impar > 0) & ref_all, (X.impar == 0) & ref_all
            else:
                on, off = X[k] > 0, ref_all
            g1 = h.r[on].groupby(h.id_sucursal[on]).agg(["mean", "var", "size"])
            g0 = h.r[off].groupby(h.id_sucursal[off]).agg(["mean", "var", "size"])
            j = g1.join(g0, lsuffix="1", rsuffix="0", how="inner")
            j = j[(j.size1 >= 3) & (j.size0 >= 5)]
            b = j.mean1 - j.mean0
            se2 = (j.var1.fillna(0) / j.size1 + j.var0.fillna(0) / j.size0).clip(lower=1e-4)
            t2 = max(0.0, float(b.var() - se2.mean()))
            dev[k] = (t2 / (t2 + se2) * b).rename(k)
            tau2[k] = t2
        self.dev = pd.concat(dev.values(), axis=1).fillna(0.0)
        self.tau2 = tau2
        # 3) nivel causal por llave y residuos un-paso-adelante
        d = self.dev.reindex(h.id_sucursal).fillna(0.0)[self.propios].values
        h["cal"] = X[bcols].values @ self.beta.values + (X[self.propios].values * d).sum(axis=1)
        h["e"] = h.y - h.cal
        self.dev_letras = None
        if self.letras and "llave_letras" in h:
            # pasada 1: residuo un-paso-adelante sin efecto de letras
            alpha0 = 1 - 0.5 ** (1 / self.h)
            h = h.sort_values(["k", "id_fecha"])
            lv0 = h.groupby("k").e.transform(lambda s: s.ewm(alpha=alpha0, adjust=True).mean().shift(1))
            u0 = (h.e - lv0)
            con = h.llave_letras.ne("SIN") & u0.notna()
            g = u0[con].groupby([h.id_sucursal[con], h.llave_letras[con]]).agg(["mean", "size"])
            w = g["size"] / (g["size"] + self.k_letras)
            self.dev_letras = (w * g["mean"]).rename("dl")
            dl = self.dev_letras.reindex(pd.MultiIndex.from_arrays([h.id_sucursal, h.llave_letras])).fillna(0.0).values
            h["e"] = h.e - dl
            h["cal"] = h.cal + dl
        med = h.groupby("k").e.transform("median")
        mad = (h.e - med).abs().groupby(h.k).transform("median") * 1.4826 + 1e-6
        h["e_w"] = h.e.clip(med - 3.5 * mad, med + 3.5 * mad)
        alpha = 1 - 0.5 ** (1 / self.h)
        h = h.sort_values(["k", "id_fecha"])
        ew = h.groupby("k").e_w.transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean())
        h["nivel_prev"] = ew.groupby(h.k).shift(1)
        h["u"] = h.e - h.nivel_prev
        self.nivel = ew.groupby(h.k).last()
        if self.regimen:
            pv = ew.groupby([h.id_sucursal, h.regimen]).last().unstack()
            self.delta_reg = float((pv["IMPAR"] - pv["PAR"]).median()) if {"IMPAR", "PAR"} <= set(pv.columns) else 0.0
        X = X.loc[h.index]
        h["estrato"] = X.estrato.values
        u = h.dropna(subset=["u"]).groupby("id_sucursal").tail(250)
        u["eu"] = np.exp(u.u.clip(upper=u.u.quantile(0.999)))
        self.smear_g = u.groupby("estrato").eu.mean()
        self.smear_s = u.groupby(["id_sucursal", "estrato"]).eu.mean()
        self.qg = u.groupby("estrato").u.quantile(QUANTS).unstack()
        self.qs = u.groupby(["id_sucursal", "estrato"]).u.quantile(QUANTS).unstack()
        self.ns = u.groupby(["id_sucursal", "estrato"]).size()
        self.n_dias = h.groupby("id_sucursal").size()
        self.n_dias_llave = h.groupby("k").size()
        return self

    def predecir(self, objetivo: pd.DataFrame) -> pd.DataFrame:
        """objetivo: una fila por sucursal con columnas de calendario del día objetivo."""
        X = variables(objetivo)
        s = objetivo.id_sucursal.values
        d = self.dev.reindex(s).fillna(0.0)[self.propios].values
        bcols = list(self.beta.index)
        cal = X[bcols].values @ self.beta.values + (X[self.propios].values * d).sum(axis=1)
        k = self._llave(objetivo).values
        niv = self.nivel.reindex(k).values
        nllave = self.n_dias_llave.reindex(k).fillna(0).values
        if self.regimen:                         # respaldo: nivel del otro régimen + diferencia típica
            otro = np.where(objetivo.regimen.values == "IMPAR", "PAR", "IMPAR")
            k2 = objetivo.id_sucursal.astype(str).values + "|" + otro
            sign = np.where(objetivo.regimen.values == "IMPAR", 1.0, -1.0)
            alt = self.nivel.reindex(k2).values + sign * self.delta_reg
            niv = np.where(np.isnan(niv) | (nllave < 10), alt, niv)
        if self.dev_letras is not None and "llave_letras" in objetivo:
            cal = cal + self.dev_letras.reindex(pd.MultiIndex.from_arrays([s, objetivo.llave_letras.values])).fillna(0.0).values
        yhat = niv + cal
        est = X.estrato.values
        idx = pd.MultiIndex.from_arrays([s, est])
        qs = self.qs.reindex(idx).values
        ns = self.ns.reindex(idx).fillna(0).values
        qg = self.qg.reindex(est).values
        w = (ns / (ns + self.n0))[:, None]
        q = np.where(np.isnan(qs), qg, w * qs + (1 - w) * qg)
        if q.ndim != 2 or q.shape[1] != len(QUANTS):
            raise RuntimeError(f"Cuantiles con forma inválida {q.shape}: qs={self.qs.shape} qg={self.qg.shape} "
                               f"estratos={self.qg.index.tolist()}")
        q = np.maximum.accumulate(q, axis=1)
        out = pd.DataFrame({"id_sucursal": s, "yhat_log": yhat, "estrato": est,
                            "n_dias": self.n_dias.reindex(s).values, "n_dias_regimen": nllave})
        for i, a in enumerate(QUANTS):
            out[f"q{int(a*100):02d}"] = np.expm1(yhat + q[:, i]).clip(min=0)
        out["p50"] = out.q50
        # media: smearing de Duan E[exp(y)] = exp(yhat)·E[exp(u)], sucursal mezclada con global
        ss = self.smear_s.reindex(idx).values
        sg = self.smear_g.reindex(est).values
        wv = w[:, 0]
        sm = np.where(np.isnan(ss), sg, wv * ss + (1 - wv) * sg)
        out["media"] = (np.exp(yhat) * sm - 1).clip(min=0)
        return out


    def explicar(self, objetivo: pd.DataFrame, top: int = 4) -> pd.Series:
        """Factores del pronóstico: contribución multiplicativa (%) de cada efecto, los `top` de mayor magnitud."""
        X = variables(objetivo)
        s = objetivo.id_sucursal.values
        bcols = list(self.beta.index)
        C = pd.DataFrame(X[bcols].values * self.beta.values, columns=bcols, index=objetivo.index)
        d = self.dev.reindex(s).fillna(0.0)[self.propios].values
        for j, k in enumerate(self.propios):
            C[k] = C.get(k, 0) + X[k].values * d[:, j]
        if self.dev_letras is not None and "llave_letras" in objetivo:
            C["letras_propias"] = self.dev_letras.reindex(pd.MultiIndex.from_arrays([s, objetivo.llave_letras.values])).fillna(0.0).values
        nombres = {"am1": "dia_letra_AM", "am2": "segundo_dia_AM", "becas1": "dia_letra_BECAS", "becas2": "segundo_dia_BECAS",
                   "stock_am_h2": "saldo_pendiente_AM_corto", "stock_am_h8": "saldo_pendiente_AM_largo",
                   "stock_becas_h2": "saldo_pendiente_BECAS_corto", "stock_becas_h8": "saldo_pendiente_BECAS_largo",
                   "log_monto_am": "monto_AM_dispersado", "evento_pe": "evento_LEEN_LCEN", "quincena": "quincena"}
        out = []
        for i in range(len(C)):
            row = C.iloc[i]
            row = row[row.abs() > 0.01]
            row = row.reindex(row.abs().sort_values(ascending=False).index)[:top]
            out.append({nombres.get(k, k): round((np.exp(v) - 1) * 100, 1) for k, v in row.items()})
        return pd.Series([__import__("json").dumps(o, ensure_ascii=False) for o in out], index=objetivo.index)


# --------------------------------------------------------------------------- baselines
def baseline_b0(hist: pd.DataFrame, objetivo: pd.DataFrame) -> pd.Series:
    """Mismo día de la semana más reciente válido."""
    h = hist[["id_sucursal", "id_fecha", "dow", "retiro"]].sort_values("id_fecha")
    last = h.groupby(["id_sucursal", "dow"]).retiro.last()
    return pd.Series(last.reindex(pd.MultiIndex.from_arrays([objetivo.id_sucursal, objetivo.dow])).values, index=objetivo.index)


def baseline_b1(hist: pd.DataFrame, objetivo: pd.DataFrame, n: int = 8) -> pd.Series:
    """Mediana de los últimos n días válidos del mismo contexto (tipo de día x régimen); respaldo: mediana 20 días."""
    hc = contexto_b1(variables(hist))
    h = hist.assign(ctx=hc.values)[["id_sucursal", "id_fecha", "ctx", "retiro"]].sort_values("id_fecha")
    med = h.groupby(["id_sucursal", "ctx"]).tail(n).groupby(["id_sucursal", "ctx"]).retiro.median()
    oc = contexto_b1(variables(objetivo))
    v = med.reindex(pd.MultiIndex.from_arrays([objetivo.id_sucursal, oc])).values
    fb = h.groupby("id_sucursal").tail(20).groupby("id_sucursal").retiro.median().reindex(objetivo.id_sucursal).values
    return pd.Series(np.where(np.isnan(v), fb, v), index=objetivo.index)


def baseline_b2(hist: pd.DataFrame, objetivo: pd.DataFrame, n: int = 20) -> pd.Series:
    h = hist.sort_values("id_fecha").groupby("id_sucursal").tail(n)
    return pd.Series(h.groupby("id_sucursal").retiro.mean().reindex(objetivo.id_sucursal).values, index=objetivo.index)


def baseline_nb(hist: pd.DataFrame, objetivo: pd.DataFrame, letras: pd.Series) -> pd.Series:
    """Método del notebook corregido (sin fuga): mediana de los últimos 3 días con las mismas letras y bandera de
    segundo día; respaldo: mediana de los últimos 6 meses (todos los contextos mezclados, como el notebook)."""
    h = hist[["id_sucursal", "id_fecha", "retiro"]].copy()
    h["llave"] = h.id_fecha.map(letras).fillna("SIN")
    h = h.sort_values("id_fecha")
    med = h.groupby(["id_sucursal", "llave"]).tail(3).groupby(["id_sucursal", "llave"]).retiro.median()
    ol = objetivo.id_fecha.map(letras).fillna("SIN")
    v = med.reindex(pd.MultiIndex.from_arrays([objetivo.id_sucursal, ol])).values
    v = np.where(ol.values == "SIN", np.nan, v)              # notebook: sin dispersión -> _hist
    t0 = pd.Timestamp(str(int(objetivo.id_fecha.iloc[0])))
    lim = int((t0 - pd.DateOffset(months=6)).strftime("%Y%m%d"))   # 6 meses calendario, como el notebook
    fb = h[h.id_fecha >= lim].groupby("id_sucursal").retiro.median().reindex(objetivo.id_sucursal).values
    return pd.Series(np.where(np.isnan(v), fb, v), index=objetivo.index)
