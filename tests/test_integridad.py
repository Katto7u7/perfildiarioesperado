"""Pruebas de integridad obligatorias (docs/03 §7). Ejecutar: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "evidencias"))

from perfil_retiro.contrato import COLS_PERFIL, validar_perfil  # noqa: E402
from perfil_retiro.evaluacion import dm_test, pinball, wape  # noqa: E402
from perfil_retiro.modelo import M1  # noqa: E402
from perfil_retiro.panel import cierres_bancarios, pascua  # noqa: E402

D, T = 20260818, 20260819


@pytest.fixture(scope="module")
def datos():
    from comun import calendario_op, panel_completo
    p = panel_completo()
    cal = calendario_op()
    sucs = np.sort(p.id_sucursal.unique())[::15][:200]            # submuestra fija para rapidez
    p = p[p.id_sucursal.isin(sucs) & (p.id_fecha >= 20250602)]
    return p, cal


def _pronostico(p, cal, inyectar_real=False):
    v = p[p.valido & (p.id_fecha <= D)].copy()
    if inyectar_real:                                           # control positivo: fuga deliberada del día T
        v = pd.concat([v, p[p.valido & (p.id_fecha == T)]])
    v["evento_pe"] = False
    obj = pd.DataFrame({"id_sucursal": v.id_sucursal.unique(), "id_fecha": T})
    cT = cal.set_index("id_fecha").loc[T]
    for c in cT.index:
        obj[c] = cT[c]
    obj["evento_pe"] = False
    m = M1().ajustar(v)
    return m.predecir(obj).set_index("id_sucursal")


def test_sin_fuga_perturbando_futuro(datos):
    p, cal = datos
    a = _pronostico(p, cal)
    p2 = p.copy()
    fut = p2.id_fecha > D
    p2.loc[fut, "retiro"] = p2.loc[fut, "retiro"] * 10 + 12345     # futuro completamente alterado
    b = _pronostico(p2, cal)
    pd.testing.assert_frame_equal(a, b)


def test_control_positivo_detecta_fuga(datos):
    p, cal = datos
    a = _pronostico(p, cal)
    b = _pronostico(p, cal, inyectar_real=True)
    assert not np.allclose(a.p50.values, b.p50.reindex(a.index).values), "la prueba no detectaría una fuga"


def test_reproducible(datos):
    p, cal = datos
    pd.testing.assert_frame_equal(_pronostico(p, cal), _pronostico(p, cal))


def test_cierres_bancarios():
    assert pascua(2026).isoformat() == "2026-04-05"
    cb = cierres_bancarios([2024, 2025, 2026])
    for f in (20240328, 20240329, 20250417, 20250418, 20260402, 20260403, 20251212, 20261102):
        assert f in cb


def test_contrato_rechaza_y_no_fuerza():
    base = {c: 1.0 for c in COLS_PERFIL}
    base.update({"id_politica_run": "X", "id_sucursal": 1, "dias_observados": 100, "hora_pico_consumo": 900,
                 "intervalo_pico": "0900-0951", "tipo_perfil": "ESTABLE", "created_at_utc": "t",
                 "retiro_promedio_diario": 10, "retiro_p90_diario": 20, "retiro_p95_diario": 30, "retiro_max_diario": 40,
                 "velocidad_consumo_promedio": 1, "velocidad_consumo_p90": 2, "pct_dias_saldo_critico": 0.1,
                 "participacion_intervalo_pico": 0.3, "pct_dias_consumo_extremo": 0.0})
    malas = [dict(base, id_sucursal=2, retiro_p90_diario=5),           # p90 < promedio
             dict(base, id_sucursal=3, hora_pico_consumo=975),         # HHMM inválido
             dict(base, id_sucursal=4, tipo_perfil="MODERADO"),        # categoría fuera del CHECK
             dict(base, id_sucursal=5, score_predictibilidad=np.nan)]  # NOT NULL
    ok, rech = validar_perfil(pd.DataFrame([base] + malas))
    assert list(ok.id_sucursal) == [1]
    assert set(rech.id_sucursal) == {2, 3, 4, 5}


def test_metricas():
    y = np.array([10.0, 20.0, 30.0])
    assert wape(y, y) == 0
    assert np.isclose(wape(np.array([12.0, 18.0, 30.0]), y), 4 / 60)
    assert np.isclose(pinball(np.array([10.0]), np.array([20.0]), 0.9), 9.0)
    s, pv = dm_test(pd.Series(np.r_[np.ones(50), np.ones(50) * 1.0]), pd.Series(np.r_[np.ones(50), np.ones(50) * 1.0] + 0.5 + np.random.default_rng(0).normal(0, 0.01, 100)))
    assert pv < 0.001
