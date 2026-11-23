"""Pruebas de las correcciones de la versión 1.1.0 (revisión de producción 2026-11-23). Sin red ni bases productivas."""
import datetime as dt
import sqlite3
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from perfil_retiro import persistencia  # noqa: E402
from perfil_retiro.modelo import QUANTS  # noqa: E402
from perfil_retiro.perfil import horario_verano_eua  # noqa: E402
from perfil_retiro.pipeline import _conformal_un_origen, cobertura_monitoreo_1700, fecha_entrenamiento_m2  # noqa: E402


# ---------------------------------------------------------------- M2: corte de entrenamiento determinista y sin fuga
def test_m2_corte_es_primer_operativo_del_mes_y_nunca_futuro():
    ops = pd.Series([20261030, 20261103, 20261104, 20261105, 20261113, 20261117, 20261119, 20261120, 20261201])
    for fc in (20261103, 20261113, 20261119, 20261120):
        f = fecha_entrenamiento_m2(ops, fc)
        assert f == 20261103 and f <= fc            # mismo modelo para todo noviembre, sin importar el orden de corridas
    assert fecha_entrenamiento_m2(ops, 20261030) == 20261030
    assert fecha_entrenamiento_m2(ops, 20261201) == 20261201


# ---------------------------------------------------------------- compuerta MONITOREO@1700
def test_cobertura_monitoreo():
    dia = pd.DataFrame({"id_sucursal": [1, 2, 3, 4], "retiro": [10.0, 5.0, 0.0, 7.0]})
    assert cobertura_monitoreo_1700(dia, pd.DataFrame({"id_sucursal": [1, 2, 4]})) == 1.0
    assert cobertura_monitoreo_1700(dia, pd.DataFrame({"id_sucursal": [1]})) == pytest.approx(1 / 3)
    assert cobertura_monitoreo_1700(dia, pd.DataFrame({"id_sucursal": []})) == 0.0


# ---------------------------------------------------------------- respaldos: retención y espacio
def _db(tmp_path):
    p = tmp_path / "plan.db"
    c = sqlite3.connect(p)
    c.execute("create table t(x)")
    c.execute("insert into t values (1)")
    c.commit()
    c.close()
    return p


def test_respaldo_retencion_solo_borra_los_propios(tmp_path):
    p = _db(tmp_path)
    ajeno = tmp_path / "plan.db.bak_20261101"                         # respaldo de OTRO proceso: no se toca
    ajeno.write_text("x")
    for i in range(1, 8):
        persistencia.respaldar(p, conservar=5, hoy=dt.date(2026, 11, i))
    quedan = [x.name for x in persistencia.respaldos_existentes(p)]
    assert quedan == [f"plan.db.bak_perfil_202611{d:02d}" for d in range(3, 8)]
    assert ajeno.exists()
    assert persistencia.respaldar(p, conservar=5, hoy=dt.date(2026, 11, 7)) is None   # uno por día
    c = sqlite3.connect(tmp_path / "plan.db.bak_perfil_20261107")
    assert c.execute("select x from t").fetchone()[0] == 1               # respaldo legible y completo
    c.close()


def test_respaldo_sin_espacio_no_escribe(tmp_path, monkeypatch):
    p = _db(tmp_path)
    Uso = namedtuple("Uso", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda _: Uso(10, 10, 0))
    with pytest.raises(RuntimeError, match="Espacio libre insuficiente"):
        persistencia.respaldar(p, hoy=dt.date(2026, 11, 23))
    assert not persistencia.respaldos_existentes(p)


# ---------------------------------------------------------------- horario de verano (Baja California)
@pytest.mark.parametrize("f,esperado", [(20260307, False), (20260308, True), (20261031, True), (20261101, False),
                                        (20261105, False), (20250309, True), (20251102, False), (20250101, False)])
def test_horario_verano_eua(f, esperado):
    assert horario_verano_eua(f) is esperado


# ---------------------------------------------------------------- conformal = método validado (últimos 60,000)
def test_conformal_usa_errores_recientes():
    rng = np.random.default_rng(0)
    n_viejo, n_nuevo = 30000, 60000
    viejos = pd.DataFrame({"id_fecha": 20250101, "err": rng.normal(2.0, 0.1, n_viejo)})     # régimen viejo, muy distinto
    nuevos = pd.DataFrame({"id_fecha": 20261101, "err": rng.normal(0.0, 0.1, n_nuevo)})
    e = pd.concat([nuevos, viejos], ignore_index=True)                                       # desordenado a propósito
    e["id_sucursal"] = np.arange(len(e)) % 5000 + 10_000                                       # sucursales ajenas al objetivo
    e["estrato"] = "NO_AM"
    punto = pd.DataFrame({"id_sucursal": [1], "estrato": ["NO_AM"], "punto": [1000.0]})
    q = _conformal_un_origen(punto, e)
    assert q.q50.iloc[0] == pytest.approx(1000.0, rel=0.05)          # con todos los errores daría ~ e^0.67 x 1000
    cols = [f"q{int(a*100):02d}" for a in QUANTS]
    assert np.all(np.diff(q[cols].values[0]) >= 0)                    # cuantiles monótonos


# ---------------------------------------------------------------- monitoreo lee la base DESTINO por defecto
def test_monitoreo_usa_destino_por_defecto(monkeypatch):
    from perfil_retiro import monitoreo
    from perfil_retiro.config import cargar_config
    vistos = []

    def falso_connect(ruta, *a, **k):
        vistos.append(ruta)
        raise RuntimeError("alto")
    monkeypatch.setattr(monitoreo.sqlite3, "connect", falso_connect)
    with pytest.raises(RuntimeError, match="alto"):
        monitoreo.main([])
    destino = persistencia.ruta_destino(cargar_config()).as_posix()
    assert vistos and destino in vistos[0] and "mode=ro" in vistos[0]


# ---------------------------------------------------------------- cómputo verificado (EV-16)
def _resultado(valor):
    perf = pd.DataFrame({"id_sucursal": [1, 2], "x": [1.0, valor], "created_at_utc": ["t1", "t2"]})
    return {"run": {"id_politica_run": "PRS-X", "fecha_corte": 20261113, "mensaje": "m", "duracion_seg": valor,
                    "n_perfiles": 2, "n_exclusiones": 0, "n_predicciones": 0},
            "perf_ok": perf, "corte_ok": pd.DataFrame({"id_sucursal": [1], "hora_inicio": [900], "hora_fin": [951]}),
            "excl": pd.DataFrame({"id_sucursal": [], "motivo": []}), "pred": None, "suc": None}


def _verificar(monkeypatch, tmp_path, secuencia):
    from perfil_retiro import pipeline
    from perfil_retiro.config import cargar_config
    it = iter(secuencia)

    def falso_calcular(*a, **k):
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return _resultado(v)
    escritos = []
    monkeypatch.setattr(pipeline, "_calcular", falso_calcular)
    monkeypatch.setattr(pipeline, "conectar_dev", lambda cfg, p: sqlite3.connect(":memory:"))
    monkeypatch.setattr(pipeline, "escribir", lambda con, run, perf, *a, **k: escritos.append(perf.x.tolist()) or "ESCRITO")
    run = pipeline.correr_diario(cargar_config(), 20261113, db_path=tmp_path / "t.db")
    return run, escritos


def test_huella_resultado_ignora_tiempos_y_detecta_cambios():
    from perfil_retiro.pipeline import huella_resultado
    a, b = _resultado(2.0), _resultado(2.0)
    b["perf_ok"]["created_at_utc"] = "otro"
    b["run"]["duracion_seg"] = 99
    assert huella_resultado(a) == huella_resultado(b)
    assert huella_resultado(a) != huella_resultado(_resultado(2.0000001))


def test_verificacion_dos_iguales_publica(monkeypatch, tmp_path):
    run, esc = _verificar(monkeypatch, tmp_path, [2.0, 2.0])
    assert esc == [[1.0, 2.0]] and "2 de 2" in run["mensaje"]


def test_verificacion_discrepancia_decide_el_tercero(monkeypatch, tmp_path):
    run, esc = _verificar(monkeypatch, tmp_path, [9.0, 2.0, 2.0])
    assert esc == [[1.0, 2.0]] and "2 de 3" in run["mensaje"]


def test_verificacion_falla_y_luego_dos_iguales(monkeypatch, tmp_path):
    run, esc = _verificar(monkeypatch, tmp_path, [RuntimeError("cuantiles vacíos"), 2.0, 2.0])
    assert esc == [[1.0, 2.0]] and "1 fallidos" in run["mensaje"]


def test_verificacion_sin_coincidencia_no_publica(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="no verificado"):
        _verificar(monkeypatch, tmp_path, [1.0, 2.0, 3.0])


def test_verificacion_corrida_abortada_no_reintenta(monkeypatch, tmp_path):
    from perfil_retiro.pipeline import CorridaAbortada
    with pytest.raises(CorridaAbortada):
        _verificar(monkeypatch, tmp_path, [CorridaAbortada("día parcial"), 2.0, 2.0])
