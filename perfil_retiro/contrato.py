"""Contrato de la tabla destino: replica TODOS los CHECK/NOT NULL/PK del DDL (bancoDBPlanificacion_minimo.sql).

Una fila que no cumple NO se corrige ni se fuerza: se rechaza y se registra con el motivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS_PERFIL = ["id_politica_run", "id_sucursal", "dias_observados", "retiro_promedio_diario", "retiro_p90_diario",
               "retiro_p95_diario", "retiro_max_diario", "volatilidad_retiro_diario", "dias_cobertura_promedio",
               "dias_cobertura_p90", "saldo_promedio_cierre", "pct_dias_saldo_critico", "velocidad_consumo_promedio",
               "velocidad_consumo_p90", "aceleracion_abs_p90", "indice_agresividad_consumo",
               "indice_concentracion_intradia", "hora_pico_consumo", "intervalo_pico", "participacion_intervalo_pico",
               "pct_dias_consumo_extremo", "score_riesgo_desabasto", "score_volatilidad", "score_predictibilidad",
               "tipo_perfil", "created_at_utc"]
COLS_CORTE = ["id_politica_run", "id_sucursal", "hora_inicio", "hora_fin", "dias_observados_intervalo",
              "retiro_promedio_intervalo", "retiro_p90_intervalo", "retiro_p95_intervalo", "participacion_dia",
              "created_at_utc"]
TIPOS = {"ESTABLE", "VOLATIL", "QUIEBRE_INTRADIA", "DISPERSION", "ALTO_RIESGO", "SOBREDOTADA"}


def _hhmm_ok(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.between(0, 2359) & (x % 100).between(0, 59) & (x == np.floor(x))


def validar_perfil(df: pd.DataFrame):
    reglas = {
        "NOT_NULL": df[COLS_PERFIL].notna().all(axis=1),
        "dias_observados>=1": df.dias_observados >= 1,
        "no_negativos": (df[["retiro_promedio_diario", "retiro_p90_diario", "retiro_p95_diario", "retiro_max_diario",
                             "volatilidad_retiro_diario", "dias_cobertura_promedio", "dias_cobertura_p90",
                             "saldo_promedio_cierre", "velocidad_consumo_promedio", "velocidad_consumo_p90",
                             "aceleracion_abs_p90", "indice_agresividad_consumo", "indice_concentracion_intradia",
                             "score_riesgo_desabasto", "score_volatilidad", "score_predictibilidad"]] >= 0).all(axis=1),
        "pct_[0,1]": df[["pct_dias_saldo_critico", "participacion_intervalo_pico", "pct_dias_consumo_extremo"]].apply(
            lambda c: c.between(0, 1)).all(axis=1),
        "hora_pico_HHMM": _hhmm_ok(df.hora_pico_consumo),
        "p90>=promedio": df.retiro_p90_diario >= df.retiro_promedio_diario,
        "p95>=p90": df.retiro_p95_diario >= df.retiro_p90_diario,
        "max>=p95": df.retiro_max_diario >= df.retiro_p95_diario,
        "vel_p90>=vel_prom": df.velocidad_consumo_p90 >= df.velocidad_consumo_promedio,
        "tipo_perfil_valido": df.tipo_perfil.isin(TIPOS),
        "intervalo_pico_texto": df.intervalo_pico.astype(str).str.len() > 0,
    }
    ok = pd.concat(reglas, axis=1).fillna(False)
    falla = ok.apply(lambda r: ",".join([k for k, v in r.items() if not v]), axis=1)
    valid = ok.all(axis=1)
    dup = df.duplicated(["id_politica_run", "id_sucursal"], keep=False)
    valid &= ~dup
    falla = np.where(dup, falla + ",PK_DUPLICADA", falla)
    return df[valid].copy(), pd.DataFrame({"id_sucursal": df.id_sucursal[~valid].values,
                                           "motivo": "CONTRATO", "detalle": pd.Series(falla)[~valid.values].values})


def validar_corte(df: pd.DataFrame, sucursales_validas: set):
    reglas = {
        "NOT_NULL": df[COLS_CORTE].notna().all(axis=1),
        "hora_inicio_HHMM": _hhmm_ok(df.hora_inicio),
        "hora_fin_HHMM": _hhmm_ok(df.hora_fin) & (df.hora_fin > df.hora_inicio),
        "dias>=1": df.dias_observados_intervalo >= 1,
        "no_negativos": (df[["retiro_promedio_intervalo", "retiro_p90_intervalo", "retiro_p95_intervalo"]] >= 0).all(axis=1),
        "participacion_[0,1]": df.participacion_dia.between(0, 1),
        "p95>=p90": df.retiro_p95_intervalo >= df.retiro_p90_intervalo,
        "p90>=promedio": df.retiro_p90_intervalo >= df.retiro_promedio_intervalo,
        "FK_padre": df.id_sucursal.isin(sucursales_validas),
    }
    ok = pd.concat(reglas, axis=1).fillna(False)
    valid = ok.all(axis=1) & ~df.duplicated(["id_politica_run", "id_sucursal", "hora_inicio", "hora_fin"], keep=False)
    falla = ok.apply(lambda r: ",".join([k for k, v in r.items() if not v]), axis=1)
    return df[valid].copy(), df[~valid].assign(motivo=falla[~valid])
