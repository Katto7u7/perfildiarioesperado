"""Simulación histórica día a día (protocolo docs/03). Cada origen D usa SOLO datos con id_fecha <= D.

Para cada origen D (día operativo) se pronostica T = siguiente día operativo del calendario publicado.
Salida: una fila por (D, sucursal) con pronósticos de todos los candidatos, cuantiles de M1, riesgo y lo realizado.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .modelo import M1, QUANTS, baseline_b0, baseline_b1, baseline_b2, baseline_nb

CORTES = [900, 1100, 1300, 1700]


def letras_por_fecha(disp: pd.DataFrame) -> pd.Series:
    d = disp[disp.id_programa.isin([1, 2])].copy()
    d["tok"] = d.id_programa.map({1: "AM", 2: "BE"}) + ":" + d.letra.fillna("?") + ("*" if True else "")
    d["tok"] = np.where(d.is_second_letterday == 1, d.tok + "2", d.tok)
    return d.groupby("id_fecha").tok.apply(lambda s: "|".join(sorted(set(s))))


def eventos_pe(pe: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Eventos LEEN/LCEN conocidos al corte (solicitud <= corte): ventanas por sucursal."""
    e = pe[pe.id_fecha_solicitud <= corte].copy()
    cod = e.evento_codigo.str.extract(r"_(\d{8})_(\d{8})")
    e["v_ini"] = pd.to_numeric(cod[0]).fillna(e.id_fecha_evento)
    e["v_fin"] = pd.to_numeric(cod[1]).fillna(e.id_fecha_evento)
    return e[["id_sucursal", "v_ini", "v_fin"]]


def marcar_evento(df: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    if ev.empty:
        return pd.Series(False, index=df.index)
    m = df[["id_sucursal", "id_fecha"]].reset_index().merge(ev, on="id_sucursal", how="inner")
    m = m[(m.id_fecha >= m.v_ini) & (m.id_fecha <= m.v_fin)]
    return df.index.isin(m["index"].unique())


def cdf_desde_cuantiles(qmat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(Y <= x) interpolando en log1p sobre la rejilla QUANTS (colas: extrapolación plana acotada)."""
    lq = np.log1p(np.maximum(qmat, 0))
    lx = np.log1p(np.maximum(x, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.interp(lx[i], lq[i], QUANTS, left=0.0, right=1.0)
    out[x <= 0] = 0.0
    return out


def forma_esperada(hist: pd.DataFrame, objetivo_tipo: pd.Series, n0: float = 15.0) -> pd.DataFrame:
    """Fracción acumulada esperada por corte: mediana sucursal x tipo de día (AM/BECAS/SIN) contraída a la global."""
    h = hist[hist.forma_valida].copy()
    h["tipo"] = np.select([h.disp_am.astype(bool), h.disp_becas.astype(bool)], ["AM", "BECAS"], "SIN")
    h = h.sort_values("id_fecha").groupby(["id_sucursal", "tipo"]).tail(60)
    cols = ["p_900", "p_1100", "p_1300"]
    ms = h.groupby(["id_sucursal", "tipo"])[cols].median()
    ns = h.groupby(["id_sucursal", "tipo"]).size()
    mg = h.groupby("tipo")[cols].median()
    idx = pd.MultiIndex.from_arrays([objetivo_tipo.index, objetivo_tipo.values])
    vs, n = ms.reindex(idx).values, ns.reindex(idx).fillna(0).values[:, None]
    vg = mg.reindex(objetivo_tipo.values).values
    w = n / (n + n0)
    v = np.where(np.isnan(vs), vg, w * vs + (1 - w) * vg)
    v = np.maximum.accumulate(np.clip(v, 0, 1), axis=1)
    return pd.DataFrame(v, index=objetivo_tipo.index, columns=cols).assign(p_1700=1.0)


def correr(panel: pd.DataFrame, cal: pd.DataFrame, disp: pd.DataFrame, pe: pd.DataFrame | None,
           fenix: pd.DataFrame | None, desde: int, hasta: int, factor_critico: float = 1.0,
           vida_media: float = 4.0, n0: float = 40.0, regimen: bool = True, cada: int = 1, log=print) -> pd.DataFrame:
    ops = cal[cal.operativo].id_fecha.sort_values().values
    sig = dict(zip(ops[:-1], ops[1:]))
    letras = letras_por_fecha(disp)
    calT = cal.set_index("id_fecha")
    valid = panel[panel.valido].sort_values(["id_sucursal", "id_fecha"])
    origenes = [d for d in ops if desde <= sig.get(d, 0) <= hasta][::cada]
    pnl = panel.set_index(["id_fecha", "id_sucursal"])
    filas = []
    for i, D in enumerate(origenes):
        t0 = time.time()
        T = int(sig[D])
        hist = valid[valid.id_fecha <= D].copy()
        hist["llave_letras"] = hist.id_fecha.map(letras).fillna("SIN")
        if pe is not None:
            ev = eventos_pe(pe, D)
            hist["evento_pe"] = marcar_evento(hist, ev)
        sucs = hist.groupby("id_sucursal").size()
        sucs = sucs[sucs >= 20].index
        obj = pd.DataFrame({"id_sucursal": sucs, "id_fecha": T})
        cT = calT.loc[T]
        for c in cT.index:
            obj[c] = cT[c]
        if pe is not None:
            obj["evento_pe"] = marcar_evento(obj, ev)
        obj["llave_letras"] = letras.get(T, "SIN")
        m = M1(vida_media=vida_media, n0_cuantil=n0, regimen=regimen).ajustar(hist[hist.id_sucursal.isin(sucs)])
        pr = m.predecir(obj)
        pr["b0"] = baseline_b0(hist, obj).values
        pr["b1"] = baseline_b1(hist, obj).values
        pr["b2"] = baseline_b2(hist, obj).values
        pr["nb"] = baseline_nb(hist, obj, letras).values
        pr["origen"], pr["id_fecha"] = D, T
        pr["evento_pe"] = obj.evento_pe.values if "evento_pe" in obj else False
        if fenix is not None:
            fx = fenix[fenix.id_fecha == T].set_index("id_sucursal").pred_max
            pr["fenix"] = fx.reindex(pr.id_sucursal).values
        # ---- riesgo de saldo (presión sin dotación adicional)
        last = hist.groupby("id_sucursal").tail(1).set_index("id_sucursal")
        s0 = pnl.loc[D].saldo_cierre.reindex(pr.id_sucursal).values if D in pnl.index.get_level_values(0) else np.full(len(pr), np.nan)
        hv = hist[hist.n_cortes == 4].sort_values("id_fecha").groupby("id_sucursal").tail(20)
        depn = {c: (hv[f"depositos_{c}"] - hv[f"dotaciones_{c}"]).groupby(hv.id_sucursal).median() for c in CORTES}
        tipo = pd.Series(np.select([bool(cT.disp_am), bool(cT.disp_becas)], ["AM", "BECAS"], "SIN"), index=pr.id_sucursal)
        tipo = pd.Series(np.where(bool(cT.disp_am), "AM", np.where(bool(cT.disp_becas), "BECAS", "SIN")), index=pr.id_sucursal.values)
        fr = forma_esperada(hist, tipo)
        U = factor_critico * pr.q90.values
        qmat = pr[[f"q{int(a*100):02d}" for a in QUANTS]].values
        pc = []
        for c in CORTES:
            dc = depn[c].reindex(pr.id_sucursal).fillna(0).values
            f = fr[f"p_{c}"].values
            x = (s0 + dc - U) / np.maximum(f, 1e-3)
            pc.append(1 - cdf_desde_cuantiles(qmat, x))
        pc = np.column_stack(pc)
        pr["saldo_ini"] = s0
        pr["umbral"] = U
        pr["p_critico"] = np.nanmax(pc, axis=1)
        pr["corte_presion"] = np.where(pc.max(axis=1) >= 0.5, np.array(CORTES)[np.argmax(pc >= 0.5, axis=1)], 0)
        for j, c in enumerate(CORTES):
            pr[f"p_crit_{c}"] = pc[:, j]
            pr[f"f_{c}"] = fr[f"p_{c}"].values
            pr[f"dep_{c}"] = depn[c].reindex(pr.id_sucursal).fillna(0).values
        # ---- realizado
        if T in pnl.index.get_level_values(0):
            r = pnl.loc[T].reindex(pr.id_sucursal)
            pr["real"] = r.retiro.values
            pr["estado_real"] = r.estado.values
            pr["forma_valida_real"] = r.forma_valida.values
            sal = []
            for c in CORTES:
                sal.append(r.sdo_inicial.values + (r[f"depositos_{c}"] - r[f"dotaciones_{c}"]).values - r[f"retiros_{c}"].values)
            smin = np.nanmin(np.column_stack(sal), axis=1)
            pr["saldo_min_sin_dot_real"] = smin
            pr["evento_critico_real"] = np.where(np.isnan(smin), np.nan, (smin < U).astype(float))
            for j, c in enumerate(CORTES):
                pr[f"p_real_{c}"] = r[f"p_{c}"].values
        filas.append(pr)
        if i % 20 == 0:
            log(f"origen {D} -> {T}: {len(pr)} sucursales, {time.time()-t0:.1f}s")
    return pd.concat(filas, ignore_index=True)
