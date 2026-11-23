"""Reglas de calidad de datos. Todas se pueden ejecutar con datos <= fecha de corte.

Evidencia de respaldo: docs/evidencias/EV-01 y EV-02.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mov_diario(mv: pd.DataFrame) -> pd.DataFrame:
    """MOVIMIENTOS agregado a (fecha, sucursal). Marca llaves duplicadas de origen."""
    g = mv.groupby(["id_fecha", "id_sucursal"], as_index=False).agg(
        retiro=("monto", "sum"), n_programas=("id_programa", "nunique"), dup=("n_filas", lambda s: bool((s > 1).any())))
    return g


def detectar_copias(dia: pd.DataFrame, umbral: float = 0.99, min_suc: int = 100) -> pd.DataFrame:
    """Fecha cuyo vector sucursal->retiro es idéntico (>= umbral) al de la fecha cargada anterior."""
    piv = dia.pivot(index="id_sucursal", columns="id_fecha", values="retiro")
    cols = list(piv.columns)
    out = []
    for a, b in zip(cols[:-1], cols[1:]):
        x, y = piv[a], piv[b]
        m = x.notna() & y.notna() & ((x > 0) | (y > 0))
        if m.sum() < min_suc:
            continue
        eq = float((np.abs(x[m] - y[m]) < 0.01).mean())
        if eq >= umbral:
            out.append({"id_fecha": int(b), "copia_de": int(a), "pct_identico": eq})
    return pd.DataFrame(out, columns=["id_fecha", "copia_de", "pct_identico"])


def detectar_parciales(dia: pd.DataFrame, mon17: pd.DataFrame, umbral_mediana: float = 0.9) -> pd.DataFrame:
    """Carga parcial de MOVIMIENTOS: la sucursal *mediana* trae < umbral de su salida en MONITOREO@1700.

    Un traspaso masivo no la dispara (afecta a pocas sucursales), una carga parcial sí (afecta a todas).
    """
    c = mon17[["id_fecha", "id_sucursal", "retiros"]].merge(dia[["id_fecha", "id_sucursal", "retiro"]],
                                                           on=["id_fecha", "id_sucursal"], how="inner")
    c = c[c.retiros > 0]
    c["r"] = c.retiro / c.retiros
    g = c.groupby("id_fecha").r.agg(r_mediana="median", r_p25=lambda s: s.quantile(0.25)).reset_index()
    g["parcial"] = g.r_mediana < umbral_mediana
    # p25 < 0.97 NO es defecto: ocurre en quincenas/fin de mes por retiros no-programa (EV-02).
    # Solo se marca sospechoso un hundimiento fuerte del cuartil bajo sin llegar a parcial.
    g["sospechoso"] = (~g.parcial) & (g.r_p25 < 0.9)
    return g


def dias_operativos_monitoreo(mon: pd.DataFrame) -> pd.DataFrame:
    """Día operativo = Σ transacciones de clientes en el corte 1700 > 0 (EV-01 §4)."""
    m = mon[mon.hora_corte == 1700].groupby("id_fecha").n_trans.sum().rename("n_trans").reset_index()
    m["operativo_mon"] = m.n_trans > 0
    return m
