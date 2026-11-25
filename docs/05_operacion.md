# Operación diaria, monitoreo y criterios de operación (versión 1.2.0)

Carpeta que se ejecuta: `VELOCIDADES/perfil_retiro_produccion/`. Intérprete: `C:/ProgramData/anaconda3/python.exe`.

## 0. Operación en sombra (decidida el 2026-11-18; 4–6 semanas)
- Script: `correr_sombra.bat`. Corre `run-daily --auto --excel` y, los viernes, `python -m perfil_retiro.monitoreo`.
  Bitácora: `out/logs/sombra.log` (inicio, fin y código de salida de cada noche) y `out/logs/perfil_retiro.log`.
- Registrar la tarea (una vez, en una terminal de Windows):
  `schtasks /Create /TN PerfilRetiroSombra /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 22:00 /TR "C:\Users\Gerardo\Desktop\Arquitectura\inversa\ANALISIS\VELOCIDADES\perfil_retiro_produccion\correr_sombra.bat"`
- Condiciones para salir de la sombra (criterios que fallaron en el holdout), medidas por `monitoreo` con ≥ 20 días objetivo:
  1. |sesgo de `retiro_esperado`| ≤ 3%;
  2. cobertura por sucursal ≥ lo esperado con calibración perfecta − 5 pp;
  3. P(crítico al cierre) con error ≤ 5 pp en todos los deciles;
  4. WAPE ≤ 1.15 × 0.352.

## 1. Ejecución nocturna
- Cuándo: días operativos ~22:00, **después** de que MOVIMIENTOS tenga el cierre del día.
- Qué produce: perfil del régimen del día objetivo (`PERFIL_RETIRO_SUCURSAL` + `_CORTE`), predicción del **siguiente día operativo** (`PREDICCION_RETIRO_DIARIA`), exclusiones y bitácora (`PERFIL_RUN`), y el Excel `out/entregables/perfil_retiro_<fecha_corte>.xlsx`.
- `--auto`: fecha de corte = último día con cierre en MOVIMIENTOS.
  - Si ese día ya tiene corrida vigente, **no se reprocesa** (registra `SIN DATOS NUEVOS` y re-exporta el Excel).
  - Si MOVIMIENTOS está atrasado respecto al calendario, registra `DATOS ATRASADOS`.
  - Para reprocesar un día a propósito: `--fecha-corte AAAAMMDD` (o `--force` para reescribir el mismo id).
- **Cómputo verificado (EV-16):** el cálculo completo se hace dos veces, desde la extracción, y solo se publica si ambos resultados coinciden bit a bit. Si no coinciden o uno falla, se hacen cómputos adicionales (máximo 5, `modelo.computos_max`) hasta que dos coincidan. Si no se logra, **no se publica nada** (código 1). Duración típica: 6–9 minutos; peor caso ~20.

## 2. Códigos de salida
| Código | Significado | Acción |
|---|---|---|
| 0 | Corrida escrita, `OMITIDO_YA_EXISTE` (mismos insumos ya procesados) o `SIN DATOS NUEVOS` | Ninguna |
| 2 | **Abortada a propósito**: día no operativo, MOVIMIENTOS sin el cierre de D, MONITOREO@1700 de D incompleto (< 90%), o D detectado como COPIA/PARCIAL | Revisar la carga del día. No se publica nada |
| 1 | Error inesperado, cómputo no verificado o falta de espacio para el respaldo | Revisar `out/logs/perfil_retiro.log`; corregir y reejecutar con la misma fecha |
| 3 | (solo `monitoreo`) hay alertas | Leer `out/monitoreo/monitoreo_AAAAMMDD.md` |

## 3. Reejecución y auditoría
- `id_politica_run = PRS-{fecha_corte}-{versión}-{hash}`. El hash cubre la configuración (sin la sección `destino`) y la huella de **todos** los insumos que pueden cambiar el resultado: MOVIMIENTOS, MONITOREO_CORTE, DISPERSION, FECHA, SUCURSAL y PROGRAMAS_EMERGENTES conocidos al corte.
- Mismos insumos → mismo id → `OMITIDO_YA_EXISTE` (no duplica). Insumos corregidos o versión nueva → id nuevo, y la corrida anterior de esa fecha queda `vigente = 0` (en SQLite y, al sincronizar, en Postgres). Nunca se borra historial.
- `PERFIL_RUN` guarda versión, hash de configuración, huella de insumos y parámetros completos: la corrida se puede reconstruir. El código es determinista, pero **esta máquina no siempre lo es**: de 5 reconstrucciones del 13/nov, 3 fueron idénticas bit a bit (una reentrenando M2 desde cero), 1 salió distinta en silencio y 1 falló (EV-16). Por eso la publicación exige cómputo verificado.
- M2 se entrena con datos hasta el primer día operativo del mes de la fecha de corte (el esquema validado en el backtest). Reprocesar cualquier fecha nunca usa datos posteriores a ella.

## 3.1 Origen de las fuentes (`.env`)
- `FUENTE=sqlite` (por defecto) o `FUENTE=postgres` en el `.env` de la raíz (plantilla: `.env.example`). Con Postgres se usan `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` de ese archivo, con una sesión de solo lectura. No afecta el destino, que sigue usando `kilometraje/.env`.
- Tablas leídas: `sucursal.movimientos`, `.fecha`, `.dispersion`, `.sucursal`, `saldos_monitoreo.monitoreo_corte` y `dboperativo_u.programas_emergentes`.
- `DISPERSION` se consolida a una fila por fecha y programa; las filas repetidas cuentan una sola vez.
- Al cambiar de fuente cambia la huella de insumos y, por lo tanto, `id_politica_run`.
- **Datos observados en Postgres el 2026-11-25** (no corregidos por el código): MOVIMIENTOS duplicado en 20260324 y 20260821 (esas filas quedan `DUPLICADO` y se excluyen) e incompleto en 20260114 y 20260305. La detección de cargas parciales solo revisa el día de corte: corregirlos antes de producir desde Postgres. `check-sources` los lista.

## 4. Escrituras
- **Destino:** `C:/Users/Gerardo/Desktop/DATABASES_LEGACY/bancoDBPlanificacion_minimo.db` (autorizado por el usuario el 2026-11-20).
  - Se escribe en `PERFIL_RETIRO_SUCURSAL` y `_CORTE`, que ya existían. Si faltan, la corrida aborta: no se crean.
  - Se crean solo si faltan `PREDICCION_RETIRO_DIARIA`, `PERFIL_RUN` y `PERFIL_EXCLUSION`.
  - El catálogo `SUCURSAL` nunca se modifica: lo que no exista ahí se excluye con el motivo `FK_SUCURSAL`.
  - Todo va en una sola transacción: o se escribe completa la corrida, o no se escribe nada.
- **Respaldo** antes de la primera escritura de cada día: `bancoDBPlanificacion_minimo.db.bak_perfil_AAAAMMDD`, hecho con el API de backup de SQLite.
  - Antes verifica que haya espacio libre ≥ 1.5 × el tamaño de la base + 1 GB; si no hay, no escribe.
  - Conserva los 5 respaldos más recientes (`destino.respaldos_a_conservar`) y solo borra sus propios `.bak_perfil_*`.
- **Postgres** (base `sucursales`, esquema `planeacion`, usuario `appserver`, credenciales en `kilometraje/.env`). Encendido desde el 2026-11-23 (`destino.postgres.habilitado: true`).
  - Cada `run-daily` vuelca todas las corridas con `pg_sincronizado = 0`.
  - Antes de insertar verifica la FK con `planeacion.sucursal` y da un error claro si falta alguna sucursal.
  - Compara los conteos por tabla; ante cualquier diferencia hace ROLLBACK total.
  - Propaga la vigencia de las corridas reemplazadas.
  - Si Postgres falla, la corrida diaria no falla: lo registra y lo reintenta la noche siguiente.
  - Prueba sin cambios: `python -m perfil_retiro.cli sync-postgres --dry-run`.
- **Estructura en el servidor:** la crea el DBA con `sql/postgres/01_planeacion_perfil_retiro.sql` y los permisos con `02_grants_appserver.sql` (aplicados el 2026-11-23).
- **Seguridad:** `kilometraje/.env.example` contiene credenciales reales. Recomendación: dejar ahí solo valores de ejemplo y rotar la contraseña.

## 5. Monitoreo (`python -m perfil_retiro.monitoreo`; los viernes desde el .bat)
Lee la base destino en solo lectura. Lo realizado usa las mismas reglas de validez que el pipeline. **Señales implementadas:**

| Señal | Cálculo | Alerta |
|---|---|---|
| Error del pronóstico | WAPE de p50 en los últimos 20 días objetivo realizados | > 1.15 × 0.352 |
| Sesgo | Σ(retiro_esperado − real)/Σreal | \|sesgo\| > 3% con ≥ 10 días |
| Cobertura P10–P90 | global | fuera de [72%, 88%] con ≥ 5 días |
| Cobertura por sucursal | % de sucursales con cobertura en [70%, 90%] vs lo esperado por binomial | < esperado − 5 pp |
| Calibración de riesgo | error máximo por decil de P(crítico al cierre) y P(quiebre intradía); AUC | se reporta en los criterios de salida |
| Atraso | días operativos posteriores a la última fecha de corte | > 1 |
| Estabilidad | % de sucursales que cambian de `tipo_perfil` vs la corrida anterior del mismo régimen | > 20% (backtest: 9–14%) |
| Cobertura del perfil | cambio de `n_perfiles` vs la corrida anterior del mismo régimen | > 3% |
| Confianza | % BAJA + INSUFICIENTE vs la corrida anterior | sube > 5 pp |

No están automatizados (se revisan a mano en la recalibración mensual): validez de categorías (EV-14) y la razón MOVIMIENTOS/MONITOREO por día.

## 6. Criterios de uso de los resultados
- Una predicción con `nivel_confianza` BAJA o INSUFICIENTE **no debe usarse sola**: hay que revisarla con el equipo operativo.
- Las probabilidades de saldo suponen que **no hay dotación adicional** en el día.
- `tipo_perfil` siempre se lee junto con su régimen (IMPAR/PAR).
- `retiro_esperado` (la media para sumar entre sucursales) tuvo sesgo +6% en el holdout: hasta salir de la sombra, para decisiones conviene usar p50/p90.
- Diciembre y Semana Santa tienen **un solo antecedente** en la historia útil: en esos periodos los intervalos son menos confiables.

## 7. Mantenimiento
- Recalibración mensual: rerun de EV-12 sobre las predicciones realizadas y de EV-14. Cualquier cambio de umbrales se registra en `config.yaml` y en el protocolo §8.
- Cambio de versión del modelo: subir `version_modelo` y correr el backtest antes de habilitarlo.
- Regenerar la semilla de calibración: `python evidencias/generar_semilla.py`, que requiere el backtest calibrado.
- Datos que mejorarían el sistema: lista oficial de sucursales que abren a las 8:00 (hoy inferida); hora exacta de extracción de cada corte; cierre real de los eventos emergentes; calendario de SEMBRANDO, LEEN y otros programas.

## 8. Registro de cambios
- **1.2.0 (2026-11-25).** Cambia el modelo: `DISPERSION` queda solo como calendario y el tamaño de cada pago se mide con retiro real (docs/02). Fuentes seleccionables por `.env` (§3.1). La semilla `out/semilla_errores.parquet` sigue siendo la de 1.1.0: no programar esta versión en la corrida nocturna sin revisar el backtest comparativo y regenerar la semilla.
- **1.1.0 (2026-11-23), revisión de producción.** El modelo estadístico no cambia.
  1. Cómputo verificado (doble cómputo bit a bit, EV-16).
  2. M2 entrenado en cada cómputo con un corte determinista (primer día operativo del mes, como en el backtest). Antes dependía de qué corrida del mes se ejecutara primero, y reprocesar un día anterior podía usar datos futuros.
  3. Conformal idéntico al validado: cuantiles globales con los últimos 60,000 errores por estrato (antes, todos).
  4. Compuerta MONITOREO@1700 ≥ 90% de las sucursales con retiro.
  5. Aviso cuando DISPERSION no está cargada hasta el día objetivo.
  6. Huella con SUCURSAL y PROGRAMAS_EMERGENTES.
  7. Horario de verano de Baja California exacto (2.º domingo de marzo a 1.er domingo de noviembre).
  8. Respaldos con retención y verificación de espacio.
  9. Postgres: verificación de FK y propagación de la vigencia.
  10. Intervalos rechazados por el contrato registrados en `PERFIL_EXCLUSION` (`CONTRATO_INTERVALO`).
  11. `monitoreo` lee la base destino (antes leía `out/dev.db`), usa el panel limpio y evalúa los 4 criterios de salida.
  12. `--auto` no reprocesa una fecha que ya tiene corrida vigente.
  13. Probabilidades vacías cuando no hay saldo inicial.
- **1.0.0-dev (2026-11-18 a 2026-11-20):** versión validada en desarrollo y holdout (EV-12, EV-13, EV-15).
