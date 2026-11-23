"""Calendario operativo y panel diario limpio (sucursal x día operativo).

Anti-fuga: `construir_panel` recibe datos ya acotados a <= fecha_corte; el calendario
(FECHA + reglas de cierre bancario + DISPERSION) es información publicada por adelantado.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .calidad import mov_diario

CORTES = [900, 1100, 1300, 1700]


# --------------------------------------------------------------------------- calendario
def pascua(anio: int) -> dt.date:
    """Domingo de Pascua (algoritmo gregoriano anónimo)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mes = (h + l_ - 7 * m + 114) // 31
    dia = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(anio, mes, dia)


def cierres_bancarios(anios) -> dict[int, str]:
    """Cierres bancarios no marcados en FECHA.is_holiday (verificados en EV-01 §4)."""
    out = {}
    for y in anios:
        p = pascua(y)
        for delta, nom in ((-3, "JUEVES_SANTO"), (-2, "VIERNES_SANTO")):
            d = p + dt.timedelta(days=delta)
            out[int(d.strftime("%Y%m%d"))] = nom
        out[y * 10000 + 1212] = "12_DICIEMBRE"
        out[y * 10000 + 1102] = "2_NOVIEMBRE"
    return out


def calendario_operativo(fecha: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    """Calendario con día operativo y variables de contexto conocidas por adelantado."""
    c = fecha[["id_fecha", "dia", "mes", "anio", "is_holiday"]].copy().sort_values("id_fecha")
    c["fecha"] = pd.to_datetime(c.id_fecha.astype(str), format="%Y%m%d")
    c["dow"] = c.fecha.dt.dayofweek
    cb = cierres_bancarios(range(int(c.anio.min()), int(c.anio.max()) + 1))
    c["cierre_bancario"] = c.id_fecha.map(cb)
    c["operativo"] = (c.dow < 5) & (c.is_holiday == 0) & c.cierre_bancario.isna()
    c["regimen"] = np.where(c.mes % 2 == 1, "IMPAR", "PAR")

    op = c[c.operativo].copy()
    op["ym"] = op.anio * 100 + op.mes
    op["dia_habil_mes"] = op.groupby("ym").cumcount() + 1
    op["n_habiles_mes"] = op.groupby("ym").id_fecha.transform("size")
    op["dias_al_fin_mes"] = op.n_habiles_mes - op.dia_habil_mes            # 0 = último hábil
    q1 = op[op.dia <= 15].groupby("ym").id_fecha.max()
    op["es_quincena"] = op.id_fecha.isin(q1.values) | (op.dias_al_fin_mes == 0)
    # días naturales desde el día operativo anterior (lunes = 3, post-feriado >= 2)
    op["gap_previo"] = op.fecha.diff().dt.days.fillna(1).astype(int)
    op["gap_siguiente"] = (-op.fecha.diff(-1)).dt.days.fillna(1).astype(int)
    op["post_cierre"] = op.gap_previo > 1                                   # lunes o post feriado
    op["post_feriado"] = (op.gap_previo > 1) & ~((op.dow == 0) & (op.gap_previo == 3))
    op["pre_cierre"] = op.gap_siguiente > 1
    op["pre_feriado"] = (op.gap_siguiente > 1) & ~((op.dow == 4) & (op.gap_siguiente == 3))
    op["pos_quincena"] = op.es_quincena.shift(1, fill_value=False)
    op = op.drop(columns=["ym"])

    # dispersión: monto/beneficiarios por fecha y programa (repetidos por letra -> max)
    d = disp[disp.id_programa.isin([1, 2])].copy()
    agg = d.groupby(["id_fecha", "id_programa"]).agg(
        n_letras=("letra", "nunique"), segundo=("is_second_letterday", "max"),
        primero=("is_second_letterday", lambda s: int((s == 0).any())),
        monto=("monto", "max"), benef=("numero_beneficiarios", "max")).reset_index()
    for pid, nom in ((1, "am"), (2, "becas")):
        a = agg[agg.id_programa == pid].set_index("id_fecha")
        op[f"disp_{nom}"] = op.id_fecha.isin(a.index)
        op[f"disp_{nom}_segundo"] = op.id_fecha.map(a.segundo).fillna(0).astype(int).astype(bool)
        op[f"disp_{nom}_primero"] = op.id_fecha.map(a.primero).fillna(0).astype(int).astype(bool)
        op[f"disp_{nom}_monto"] = op.id_fecha.map(a.monto).fillna(0.0)
        op[f"disp_{nom}_benef"] = op.id_fecha.map(a.benef).fillna(0.0)
        op[f"disp_{nom}_letras"] = op.id_fecha.map(a.n_letras).fillna(0).astype(int)
    # "Saldo pendiente de retirar" de dispersiones ANTERIORES (no incluye el día): Σ monto(d') λ^(Δ días hábiles - 1).
    # Solo el primer día de cada letra aporta monto (el segundo día repite el mismo monto en DISPERSION).
    op = op.sort_values("id_fecha").reset_index(drop=True)
    for nom in ("am", "becas"):
        aporte = np.where(op[f"disp_{nom}_primero"], op[f"disp_{nom}_monto"], 0.0) / 1e9
        for vm in (2, 8):
            lam = 0.5 ** (1 / vm)
            st = np.zeros(len(op))
            acc = 0.0
            for i in range(len(op)):
                st[i] = acc                       # stock antes del día i (solo días previos)
                acc = acc * lam + aporte[i]
            op[f"stock_{nom}_h{vm}"] = np.log1p(st)
    otros = set(disp[~disp.id_programa.isin([1, 2])].id_fecha)
    op["disp_otro"] = op.id_fecha.isin(otros)
    op["disp_alguna"] = op.disp_am | op.disp_becas | op.disp_otro
    return c.merge(op.drop(columns=["dia", "mes", "anio", "is_holiday", "fecha", "dow", "cierre_bancario", "operativo", "regimen"]),
                   on="id_fecha", how="left")


# --------------------------------------------------------------------------- intradía
def intradia(mon: pd.DataFrame) -> pd.DataFrame:
    """Pivotea cortes acumulados a una fila por sucursal-día (retiro acumulado y saldos por corte)."""
    m = mon[mon.hora_corte.isin(CORTES)]
    w = m.pivot_table(index=["id_fecha", "id_sucursal"], columns="hora_corte",
                      values=["retiros", "sdo_final", "dotaciones", "depositos"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    ini = m[m.hora_corte == 900][["id_fecha", "id_sucursal", "sdo_inicial"]]
    t17 = m[m.hora_corte == 1700][["id_fecha", "id_sucursal", "n_trans", "in_service"]]
    w = w.merge(ini, on=["id_fecha", "id_sucursal"], how="left").merge(t17, on=["id_fecha", "id_sucursal"], how="left")
    w["n_cortes"] = w[[f"retiros_{c}" for c in CORTES]].notna().sum(axis=1)
    return w


# --------------------------------------------------------------------------- panel
def construir_panel(mv: pd.DataFrame, mon: pd.DataFrame, cal: pd.DataFrame, cfg, fecha_corte: int,
                    copias=(), parciales=(), sospechosos=(), desde: int | None = None) -> pd.DataFrame:
    """Panel sucursal x día operativo en [desde|inicio_util, fecha_corte]. No imputa: marca estado del dato."""
    ini = int(desde if desde is not None else cfg["historia"]["inicio_util"])
    tol, tol_abs = cfg["calidad"]["tolerancia_traspaso"], cfg["calidad"]["tolerancia_traspaso_abs"]
    mv = mv[(mv.id_fecha >= ini) & (mv.id_fecha <= fecha_corte)]
    mon = mon[(mon.id_fecha >= ini) & (mon.id_fecha <= fecha_corte)]
    dia = mov_diario(mv)
    cargadas = set(dia.loc[dia.retiro > 0, "id_fecha"])
    ops = cal[(cal.operativo) & (cal.id_fecha >= ini) & (cal.id_fecha <= fecha_corte)]

    it = intradia(mon)
    sucs = np.union1d(dia.id_sucursal.unique(), it.id_sucursal.unique())
    grid = pd.MultiIndex.from_product([ops.id_fecha.values, sucs], names=["id_fecha", "id_sucursal"]).to_frame(index=False)
    p = grid.merge(dia, on=["id_fecha", "id_sucursal"], how="left").merge(it, on=["id_fecha", "id_sucursal"], how="left")
    p["fila_mov"] = p.retiro.notna()
    p.loc[p.retiro.isna() & p.id_fecha.isin(cargadas), "retiro"] = 0.0      # era dispersa: sin fila = 0
    p["dup"] = p.dup.astype("boolean").fillna(False).astype(bool)

    abierta = (p.n_trans.fillna(0) > 0) | (p.retiro.fillna(0) > 0)
    sin_evid = p.n_trans.isna() & ~(p.retiro.fillna(0) > 0)
    estado = np.select(
        [~p.id_fecha.isin(cargadas), p.id_fecha.isin(copias), p.id_fecha.isin(parciales), p.id_fecha.isin(sospechosos),
         sin_evid, ~abierta, p.dup],
        ["FALTANTE_CARGA", "COPIA", "PARCIAL", "SOSPECHOSO", "SIN_EVIDENCIA", "SUCURSAL_CERRADA", "DUPLICADO"],
        default="OK")
    p["estado"] = estado
    p["valido"] = p.estado == "OK"

    # forma intradía válida: 4 cortes, monótona, sin traspaso, retiro > 0
    r = p[[f"retiros_{c}" for c in CORTES]].to_numpy(float)
    mono = np.all(np.diff(r, axis=1) >= -1.0, axis=1)
    traspaso = (p.retiros_1700 - p.retiro) > np.maximum(tol * p.retiro, tol_abs)
    p["traspaso"] = traspaso.fillna(False)
    p["forma_valida"] = p.valido & (p.n_cortes == 4) & mono & ~p.traspaso & (p.retiro > 0) & (p.retiros_1700 > 0)
    for c in CORTES:
        p[f"p_{c}"] = np.where(p.forma_valida, p[f"retiros_{c}"] / p.retiros_1700, np.nan)
    # saldos
    sf = p[[f"sdo_final_{c}" for c in CORTES]]
    p["saldo_cierre"] = p.sdo_final_1700
    p["saldo_min_cortes"] = sf.min(axis=1)
    p["saldo_valido"] = p.valido & (p.n_cortes == 4) & (p.saldo_cierre >= 0) & (p.sdo_inicial >= 0)
    p["dotacion"] = p.dotaciones_1700
    p["depositos_netos"] = p.depositos_1700 - p.dotaciones_1700
    p = p.merge(cal[cal.operativo].drop(columns=["operativo", "is_holiday", "cierre_bancario"]), on="id_fecha", how="left")
    return p
