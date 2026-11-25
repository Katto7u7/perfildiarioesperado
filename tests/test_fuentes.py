"""Selección de fuente (.env) y dispersión agrupada. Sin red ni bases productivas."""
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from perfil_retiro import sources  # noqa: E402
from perfil_retiro.backtest import letras_por_fecha  # noqa: E402
from perfil_retiro.calidad import fechas_anomalas  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    monkeypatch.setattr(sources, "RUTA_ENV", p)
    monkeypatch.delenv("FUENTE", raising=False)
    return p


def test_sin_env_usa_sqlite(env):
    assert sources.fuente_activa() == "sqlite"


def test_env_elige_postgres_y_la_variable_de_proceso_gana(env, monkeypatch):
    env.write_text("FUENTE=postgres\n", encoding="utf-8")
    assert sources.fuente_activa() == "postgres"
    monkeypatch.setenv("FUENTE", "sqlite")
    assert sources.fuente_activa() == "sqlite"


def test_fuente_desconocida_falla(env):
    env.write_text("FUENTE=mysql\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FUENTE"):
        sources.fuente_activa()


def test_credenciales_del_env_ganan_y_no_tocan_os_environ(env, monkeypatch):
    monkeypatch.setenv("PGHOST", "otro")
    env.write_text("PGHOST=del-archivo\nPGPASSWORD='x y'\n", encoding="utf-8")
    e = sources.entorno()
    assert (e["PGHOST"], e["PGPASSWORD"]) == ("del-archivo", "x y")
    import os
    assert os.environ["PGHOST"] == "otro"


def test_consulta_sin_version_postgres_da_error_claro(env):
    env.write_text("FUENTE=postgres\n", encoding="utf-8")
    from perfil_retiro.config import cargar_config
    with pytest.raises(NotImplementedError, match="saldos"):
        sources.extraer(cargar_config(), "saldos", usar_cache=False)


def _disp(filas):
    return pd.DataFrame(filas, columns=["id_fecha", "id_programa", "letra", "is_second_letterday"])


def test_agrupar_dispersion_una_fila_por_fecha_y_programa():
    raw = _disp([(20260318, 1, "O", 0), (20260318, 1, "N", 0), (20260318, 1, "N", 0), (20260318, 1, "R", 1),
                 (20260318, 2, "A", 0), (20260319, 3, None, 0)])
    g = sources.agrupar_dispersion(raw)
    assert list(zip(g.id_fecha, g.id_programa)) == [(20260318, 1), (20260318, 2), (20260319, 3)]
    am = g.iloc[0]
    assert (am.letras, am.letras_segundo, am.n_letras) == ("N,O", "R", 3)
    assert (am.primero, am.segundo) == (1, 1)
    nulo = g.iloc[2]
    assert (nulo.letras, nulo.n_letras) == ("?", 0)


def test_agrupar_dispersion_no_depende_del_orden():
    raw = _disp([(20260318, 1, "O", 0), (20260318, 1, "N", 0)])
    pd.testing.assert_frame_equal(sources.agrupar_dispersion(raw), sources.agrupar_dispersion(raw.iloc[::-1]))


def test_letras_por_fecha_con_dispersion_agrupada():
    raw = _disp([(20260318, 1, "O", 0), (20260318, 1, "N", 1), (20260318, 2, "A", 0), (20260319, 3, "Z", 0)])
    s = letras_por_fecha(sources.agrupar_dispersion(raw))
    assert s.to_dict() == {20260318: "AM:N*2|AM:O*|BE:A*"}


def test_fechas_anomalas():
    filas = []
    for f in range(20260301, 20260312):
        for s in range(100 if f != 20260305 else 10):
            filas.append((f, s, 1, 2 if f == 20260309 else 1))
    mv = pd.DataFrame(filas, columns=["id_fecha", "id_sucursal", "id_programa", "n_filas"])
    r = fechas_anomalas(mv, set(range(20260301, 20260312)))
    assert r == {"duplicadas": [20260309], "incompletas": [20260305]}
