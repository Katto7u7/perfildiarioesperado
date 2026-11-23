# Etapa 1 — Diagnóstico

Cada afirmación remite a su evidencia reproducible en `docs/evidencias/EV-xx/` (script en `evidencias/`). Todas las lecturas de base de datos fueron de solo lectura (`mode=ro`, `PRAGMA query_only`).

## 1. Inventario
| Componente | Qué es | Estado |
|---|---|---|
| `VELOCIDADES_HIST 2.ipynb` (= `v2/`) | Notebook original: perfil por corte (velocidades), `perfil_dia` + `perfil_sucursal` | Investigación. Tiene fuga y errores (§4) |
| `v2/velocidades_hist.py` | Conversión literal del notebook | Mismos errores |
| `velocidades_pipeline_confiable.py`, `EXPORT_VELOCIDADES_RUST/` | Rediseño con calidad y conciliación | Mejor base técnica. Perfil "in-sample", sin run id |
| `Velocidades_20260616/VELOCIDADES_UPDATE_1.py`, `Velocidades_20260616_sonet/` (paquete, CLI y Rust) | Flujo 2 "perfil esperado" en CSV | Fuga del mismo día; velocidad esperada mal calculada |
| `Politica_sucursal/` | Motor de `POLITICA_SUCURSAL` (Python = Rust), con validación por fill-rate | Sólido. Tiene un error de fines de semana (EV-10) |
| `mal.txt` | Revisión crítica del notebook (45 puntos) | Coincide con este diagnóstico |
| `DBs/*.db` (5 bases) + `kilometraje/...politica_visita.db` | Bases productivas exportadas a SQLite | Ver §2 |

**Ningún componente previo llena `PERFIL_RETIRO_SUCURSAL` ni genera `id_politica_run`.** La tabla y su hija `PERFIL_RETIRO_SUCURSAL_CORTE` están vacías. `POLITICA_SUCURSAL` también está vacía en las bases.

## 2. Fuentes y fuente de verdad
| Variable | Fuente de verdad | Evidencia |
|---|---|---|
| Retiro diario (demanda) | `DB11.MOVIMIENTOS`. Idéntica al peso a `RETIROS` e `INGENIERIA_VARIABLES` en el 100% de los días | EV-01 |
| Salida total de efectivo | `MONITOREO_CORTE`@1700 = `SALDOS.retiros` (100%). Incluye traspasos y salida no-programa | EV-01, EV-02 |
| Forma intradía | `MONITOREO_CORTE` 9/11/13/17, sin traspasos. Validada contra `MOVIMIENTOS_CORTE` (ρ = 0.97) | EV-04 |
| Saldos | `MONITOREO_CORTE` (`sdo_inicial`, `sdo_final` por corte); `SALDOS` solo desde 2025-12 | EV-01 |
| Día operativo | FECHA + cierres bancarios (Jueves/Viernes Santo, 12-dic, 2-nov), verificados con transacciones = 0 en 2024–2026 | EV-01 §4 |
| Calendario de pagos | `DISPERSION` (AM y BECAS por letra), publicado por adelantado | EV-06 |
| Eventos LEEN/LCEN | `PROGRAMAS_EMERGENTES` (anticipación mediana de 5 días) | EV-08 |
| Visitas / capacidad | `POLITICA_VISITA`, `tope_por_sucursal.csv` | EV-10 |
| Referencias externas | `PREDICCION_DS` (sin timestamp de generación), `FORECAST_DIARIO` (contaminado). Ninguna se usa como insumo | plan |

## 3. Calidad y cobertura
- **Historia útil: desde 2025-06-02** (EV-03). Antes, MOVIMIENTOS cubre solo una fracción de la demanda: entre 32% y 100% de los días son parciales. El quiebre coincide con el cambio al archivo diario de las 18 hrs.
- **Días dañados en el periodo útil** (EV-01):
  - parciales: 2025-09-30 y **2025-12-09** (hallazgo nuevo);
  - copia: 2026-10-21;
  - 36 llaves duplicadas;
  - **ningún día operativo sin cargar.** Los "faltantes" del diagnóstico preliminar eran cierres bancarios que FECHA no marca.
- **MONITOREO:** los fines de semana arrastran el viernes; hay traspasos masivos en 9 fechas (hasta 7,400 M en un solo día); el backfill 2025-06 → 2026-02 tiene forma real (EV-04).
- **Salida no-programa en días de pago:** de +0.5% a +0.9% sobre MOVIMIENTOS. Es significativa pero pequeña (EV-02).
- Cuarentena: el 93.3% de los sucursal-días son OK; el resto se excluye sin imputar (EV-02).

## 4. Metodología actual: fortalezas y errores
**Fortalezas que se conservan como ideas:**
- usar cortes intradía;
- condicionar por letra y ciclo de dispersión;
- banderas de calidad y conciliación (versión "confiable");
- día análogo y la guarda de no usar la predicción del mismo día (UPDATE_1);
- simulación y fill-rate (política).

**Errores, todos verificados:**
1. **Fuga de información:** `_hist` incluye el día D; los terciles de riesgo usan toda la historia; CLI/Rust usan la predicción del mismo día.
2. **`drop_duplicates` por valor** en el retiro diario: colapsa días distintos con el mismo total.
3. **Fines de semana contados como días** (arrastre del viernes). El mismo error existe en el motor de política: ≈ 200 M de saldos mínimos inflados (EV-10).
4. `aceleracion_arranque` es un monto, no una aceleración; `vel_md` mezcla unidades; `vel_esperada_pred` divide el acumulado entre las horas de un solo intervalo.
5. `pct_franja` es una diferencia de medianas, no es monótona y se recorta a 0.
6. `_3c` no filtra por programa y ordena los ciclos como texto.
7. Constantes sin evidencia: 1.2, 1.5 y 0.65.
8. Riesgo por terciles relativos: un tercio de las sucursales siempre sale VOLATIL.
9. **Ventanas que mezclan regímenes:** producen perfiles 2.4 veces más alejados de la realidad del mes siguiente (EV-09).
10. Ignora el factor dominante: el día de letra AM explica 0.165 de R², el bloque más grande (EV-06).

## 5. Riesgos identificados
- Un solo diciembre y una sola Semana Santa en la historia útil: esos efectos no se pueden estimar con precisión.
- Hora de apertura desconocida: hora pico, intervalo pico y agresividad dependen de un supuesto (EV-05).
- `PREDICCION_DS` no se puede auditar: no prueba que existiera antes del día pronosticado.
- Dumps DDL desactualizados respecto a las bases vivas.
- Sucursales cerradas o con poca historia: deben excluirse explícitamente, no rellenarse.

## 6. Información adicional aprovechable
- `DISPERSION`: monto nacional dispersado del día (elasticidad 0.26), día 1 vs día 2 de letra.
- `PROGRAMAS_EMERGENTES`: +16.5% de retiro en días de evento (EV-08).
- `POLITICA_VISITA`: G (días hasta la siguiente dotación) para el riesgo antes de la siguiente visita.
- `tope_por_sucursal.csv`: capacidad física de bóveda.
- `GESTIONADA`: hora de entrega por ruta (presión intradía antes de la llegada de la dotación).

**Datos que hacen falta (solicitud al negocio):**
1. Hora real de apertura por sucursal.
2. Cierre real de los eventos emergentes (`estatus_evento`, `monto_extra_real`).
3. DDL real de Postgres (`pg_dump -s`) para confirmar los CHECK.
