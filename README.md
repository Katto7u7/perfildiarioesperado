# perfil_retiro_produccion — Perfil de retiro y pronóstico diario por sucursal (versión 1.1.0)

Copia de producción de `prod_perfil_retiro/` (revisión del 2026-11-23). **Esta es la carpeta que debe ejecutarse.**
La guía completa (qué hace, cómo se ejecuta, condiciones, limitaciones) está en `LEEME_PERFIL_RETIRO.txt`.

## Estructura
```
config/config.yaml        parámetros versionados (rutas, ventanas, umbrales, horario por huso); sin credenciales
sql/                      consultas de extracción (:desde, :hasta = fecha de corte) y scripts para el DBA (sql/postgres/)
perfil_retiro/
  sources.py              extracción de solo lectura + huella de insumos
  calidad.py              detección de días copiados y cargas parciales
  panel.py                calendario operativo (cierres bancarios, quincenas, dispersión, saldos pendientes) + panel limpio
  modelo.py               M1 (nivel por régimen, efectos contraídos, letras propias) y métodos de referencia B0/B1/B2/NB
  m2_gbm.py               M2 gradient boosting
  backtest.py             simulación histórica día a día sin fuga (investigación)
  calibracion.py          conformal causal (intervalos) + riesgo (investigación)
  evaluacion.py           WAPE, pinball, Diebold-Mariano, bootstrap por bloques, calibración de probabilidades
  perfil.py               columnas de PERFIL_RETIRO_SUCURSAL y _CORTE (hora local)
  contrato.py             todos los CHECK/NOT NULL/PK del DDL destino
  persistencia.py         id determinista, escritura transaccional idempotente, respaldo con retención
  pipeline.py             corrida diaria con cómputo verificado (doble cómputo bit a bit)
  postgres_sync.py        volcado idempotente a Postgres (planeacion)
  monitoreo.py            reporte semanal y condiciones de salida de la sombra
  exportar.py             Excel de una corrida
  cli.py                  línea de comandos
evidencias/               scripts EV-01 … EV-16 y generar_semilla.py (cada número del reporte se regenera desde aquí)
tests/                    30 pruebas: fuga (con control positivo), reproducibilidad, contrato, cierres, destino, v1.1.0
docs/                     01 diagnóstico · 02 metodología · 03 protocolo · 04 diccionario · 05 operación · evidencias/EV-xx
out/semilla_errores.parquet   errores fuera de muestra del backtest (arranque de la calibración); sha256 en el LEEME
```

## Uso rápido
```
C:/ProgramData/anaconda3/python.exe -m pytest tests -q
C:/ProgramData/anaconda3/python.exe -m perfil_retiro.cli check-sources --auto
C:/ProgramData/anaconda3/python.exe -m perfil_retiro.cli run-daily --auto --excel
C:/ProgramData/anaconda3/python.exe -m perfil_retiro.monitoreo
```
Las bases fuente se abren en solo lectura. `run-daily` escribe en la base destino de `config.yaml` (con respaldo previo)
y, si `destino.postgres.habilitado` es `true`, vuelca a Postgres.
