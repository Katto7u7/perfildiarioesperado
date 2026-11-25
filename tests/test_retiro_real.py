"""Variables de tamaño de pago calculadas con retiro real (MOVIMIENTOS), nunca con DISPERSION. Sin bases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from perfil_retiro.modelo import variables  # noqa: E402
from perfil_retiro.panel import agregar_retiro_real, calendario_operativo  # noqa: E402
from perfil_retiro.sources import agrupar_dispersion  # noqa: E402

# días hábiles de marzo 2026
DIAS = [20260302, 20260303, 20260304, 20260305, 20260306, 20260309, 20260310, 20260311, 20260312, 20260313]


def _datos():
    fecha = pd.DataFrame({"id_fecha": DIAS, "dia": [d % 100 for d in DIAS], "mes": 3, "anio": 2026, "is_holiday": 0})
    # letra A (AM, primer día) paga el 02, 04, 06, 10 y 12; letra B (AM) el 12
    filas = [(d, 1, "A", 0) for d in (20260302, 20260304, 20260306, 20260310, 20260312)] + [(20260312, 1, "B", 0)]
    disp = agrupar_dispersion(pd.DataFrame(filas, columns=["id_fecha", "id_programa", "letra", "is_second_letterday"]))
    total = {20260302: 100.0, 20260304: 200.0, 20260306: 600.0, 20260310: 400.0, 20260312: 999.0}
    mv = pd.DataFrame([(d, s, 1, total.get(d, 10.0) * 1e7 / 10, 1) for d in DIAS for s in range(10)],
                      columns=["id_fecha", "id_sucursal", "id_programa", "monto", "n_filas"])
    return calendario_operativo(fecha, disp), disp, mv


def test_la_dispersion_no_aporta_montos():
    cal, _, _ = _datos()
    assert not [c for c in cal.columns if "monto" in c or "benef" in c or c.startswith("stock_")]


def test_referencia_es_mediana_de_los_ultimos_tres_pagos_anteriores():
    cal, disp, mv = _datos()
    r = agregar_retiro_real(cal, disp, mv).set_index("id_fecha")
    ref = r.disp_am_retiro_ref
    assert np.isnan(ref[20260302])                                  # sin antecedentes
    assert ref[20260304] == pytest.approx(100e7)
    assert ref[20260310] == pytest.approx(200e7)                    # mediana(100, 200, 600)
    assert ref[20260312] == pytest.approx(400e7)                    # A: mediana(200, 600, 400); B sin antecedentes
    assert np.isnan(ref[20260303])                                  # no es día de pago


def test_saldo_reciente_solo_usa_dias_anteriores_de_pago():
    cal, disp, mv = _datos()
    r = agregar_retiro_real(cal, disp, mv).set_index("id_fecha")
    assert r.stock_am_h2[20260302] == 0
    assert r.stock_am_h2[20260303] == pytest.approx(np.log1p(1.0))              # 100e7 / 1e9
    assert r.stock_am_h2[20260304] == pytest.approx(np.log1p(0.5 ** 0.5))       # el 03 no es pago: no aporta
    assert (r.stock_becas_h2 == 0).all()


@pytest.mark.parametrize("d", DIAS)
def test_sin_fuga_alterar_desde_d_no_cambia_d(d):
    cal, disp, mv = _datos()
    cols = ["disp_am_retiro_ref", "stock_am_h2", "stock_am_h8", "stock_becas_h2", "stock_becas_h8"]
    a = agregar_retiro_real(cal, disp, mv)
    alterado = mv.copy()
    alterado.loc[alterado.id_fecha >= d, "monto"] *= 7
    b = agregar_retiro_real(cal, disp, alterado)
    pd.testing.assert_frame_equal(a[a.id_fecha <= d][cols], b[b.id_fecha <= d][cols])
    c = agregar_retiro_real(cal, disp, mv[mv.id_fecha < d])          # sin datos desde d (caso producción: objetivo)
    pd.testing.assert_frame_equal(a[a.id_fecha <= d][cols], c[c.id_fecha <= d][cols])


def test_dias_excluidos_duplicados_o_incompletos_no_cuentan():
    cal, disp, mv = _datos()
    r = agregar_retiro_real(cal, disp, mv, excluir={20260304}).set_index("id_fecha")
    assert r.disp_am_retiro_ref[20260306] == pytest.approx(100e7)
    dup = mv.copy()
    dup.loc[dup.id_fecha == 20260304, "n_filas"] = 2
    inc = mv[~((mv.id_fecha == 20260306) & (mv.id_sucursal > 1))]
    assert agregar_retiro_real(cal, disp, dup).set_index("id_fecha").disp_am_retiro_ref[20260306] == pytest.approx(100e7)
    assert agregar_retiro_real(cal, disp, inc).set_index("id_fecha").disp_am_retiro_ref[20260310] == pytest.approx(150e7)


def test_variables_exige_las_columnas_de_retiro_real():
    cal, disp, mv = _datos()
    op = cal[cal.operativo]
    with pytest.raises(KeyError, match="agregar_retiro_real"):
        variables(op)
    r = agregar_retiro_real(cal, disp, mv)
    X = variables(r[r.operativo]).set_index(r[r.operativo].id_fecha)
    assert X.log_monto_am[20260304] == pytest.approx(np.log(100e7) - 21.0)
    assert X.log_monto_am[20260302] == 0 and X.log_monto_am[20260303] == 0
