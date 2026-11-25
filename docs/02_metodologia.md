# Etapa 2 — Metodología (versión ajustada por la evidencia EV-01 … EV-10)

Cambios respecto al plan aprobado, todos forzados por la evidencia:

| Plan original | Ahora | Por qué |
|---|---|---|
| Perfil sobre los últimos 180 días hábiles | Perfil sobre los **últimos 126 días válidos del régimen vigente** (misma paridad de mes) | EV-09: error 0.26 contra 0.63; gana 10/10 meses |
| Efectos de calendario iguales por sucursal o jerárquicos, sin especificar | **AM, BECAS y régimen por sucursal** (contraídos); **día de la semana, inicio de mes y quincena comunes** | EV-07: τ > 0 e I² alto solo en esos tres |
| "Días faltantes" de MOVIMIENTOS | No hay faltantes: son **cierres bancarios** que FECHA no marca | EV-01 §4 |
| SOBREDOTADA con el máximo de la política | SOBREDOTADA por **cobertura** mientras la política no esté vigente | EV-10 §3 |
| Curva intradía por contexto genérico | Curva por sucursal × **tipo de día (AM / BECAS / SIN)**, con bandas | EV-04 d) |
| Hora pico como dato | Hora pico **dependiente del supuesto de apertura** (marcada) | EV-05 |

## 1. Pronóstico del retiro del siguiente día operativo (M1)
Escala log: `y = log1p(retiro)`. Para la sucursal *s* en el día objetivo *d*:

```
ŷ(s,d) = nivel(s,d) + Σ_k β_k(s)·x_k(d) + Σ_j γ_j·z_j(d)
```

- **Efectos por sucursal** (contraídos hacia el global): día 1 de letra AM, día 2 de letra AM, día 1 de letra BECAS, régimen IMPAR. Estimador empírico-bayesiano:
  - β̂_k(s) = w·b_k(s) + (1−w)·β_k;
  - w = τ²_k / (τ²_k + se²_k(s));
  - τ_k se reestima en cada corrida, como en la EV-07.
- **Efectos comunes** (estimados a nivel fecha): día de la semana, días hábiles 1–3 del mes, quincena, log del retiro AM real de los pagos anteriores de las mismas letras (dentro de días AM; desde 1.2.0), día 2 de BECAS, evento emergente activo con su dosis (EV-08; queda solo si pasa la ablación).
- **Nivel** = mediana ponderada exponencial (vida media configurable; se elige en el torneo) del residuo de los últimos días válidos después de quitar el calendario. Captura la tendencia y los cambios de nivel sin reestimar todo.
- **Cuantiles (P10, P50, P90, P95):** calibración conformal sobre los residuos fuera de muestra recientes, estratificada por tipo de día (AM / no AM, porque la varianza difiere). Los residuos de la sucursal se mezclan con los de su grupo según el número de observaciones.
- **Salidas:** media, P10, P50, P90, P95, P(retiro > P95 del perfil) = probabilidad de retiro extraordinario, y ancho relativo (P90 − P10)/P50.

**Tamaño del pago (desde 1.2.0).** `DISPERSION` es solo calendario (fecha, programa, letra, primer o segundo día): sus montos son proyectados y no se leen. Con `panel.agregar_retiro_real`, usando únicamente fechas anteriores a cada día:
- retiro de referencia AM: por letra, mediana del retiro nacional AM en sus últimos 3 pagos; promedio de las letras del día;
- retiro reciente AM y BECAS: retiro nacional de los días de pago anteriores, con vida media de 2 y 8 días hábiles.

Se excluyen las fechas copia, parciales, duplicadas o incompletas.

## 2. Intradía y saldo
- **Curva de fracción acumulada por corte** (9/11/13/17): mediana de la sucursal en días válidos del mismo tipo (AM / BECAS / SIN), contraída hacia la del contexto global según el número de días. Se fuerza la monotonía. Bandas: p10/p90 empíricos de la fracción.
- **Retiro esperado por intervalo** = fracción × cuantiles del total diario.
- **Saldo** (los insumos son conocidos a las 22:00 de D):
  - saldo inicial de D+1 = `sdo_final`@1700 de D;
  - depósitos netos esperados por corte = mediana del contexto;
  - dotaciones futuras **no se suponen** ("presión sin dotación adicional").
- **P(saldo crítico en el corte c)** = P(retiro acumulado a c > saldo inicial + depósitos netos a c − umbral), donde umbral = factor × P90 del día siguiente. Se obtiene de la distribución predictiva del total, combinada con la variabilidad de la fracción. El corte del primer cruce con P ≥ 0.5 es el "momento aproximado de presión".
- **Horizonte hasta la siguiente visita (G, de POLITICA_VISITA):** retiro acumulado de 1..G días por simulación de trayectorias de residuos (bootstrap por bloques). Da P(desabasto antes de la siguiente dotación posible).

## 3. Perfil (`PERFIL_RETIRO_SUCURSAL`)
**Ventana:** días válidos del régimen del mes vigente, últimos 126. **Mínimo: 60 días válidos.** Si no se alcanza, la sucursal se excluye y queda registrada con su motivo.

| Columna | Definición | Unidad |
|---|---|---|
| dias_observados | nº de días válidos en la ventana | días |
| retiro_promedio / p90 / p95 / max_diario | media / cuantil lineal / máximo del retiro diario (MOVIMIENTOS) | $ |
| volatilidad_retiro_diario | desviación estándar (ddof = 1) | $ |
| dias_cobertura_promedio | media de `saldo_cierre(d) / P50_esperado(d+1)` | días |
| dias_cobertura_p90 | media de `saldo_cierre(d) / P90_esperado(d+1)` = cobertura ante demanda alta | días |
| saldo_promedio_cierre | media de `sdo_final`@1700 (excluye cierres negativos, que se cuentan) | $ |
| pct_dias_saldo_critico | % de días con saldo mínimo en cortes < factor × P90 esperado de ese día | [0,1] |
| velocidad_consumo_promedio / p90 | velocidades por intervalo ($/h) con el monto escalado a MOVIMIENTOS (**dependen de la apertura**) | $/h |
| aceleracion_abs_p90 | p90 de \|Δvelocidad\| / Δt entre puntos medios de intervalos consecutivos | $/h² |
| indice_agresividad_consumo | mediana de [fracción a las 11:00 ÷ fracción de tiempo transcurrida a las 11:00] | razón |
| indice_concentracion_intradia | mediana de Σ pᵢ²/tᵢ − 1 (0 = uniforme en el tiempo) | ≥ 0 |
| hora_pico_consumo / intervalo_pico / participacion_intervalo_pico | intervalo con mayor velocidad mediana y su participación mediana | HHMM / texto / [0,1] |
| pct_dias_consumo_extremo | % de días con residuo robusto (vs calendario) > 3·MAD | [0,1] |
| score_riesgo_desabasto | frecuencia de días críticos contraída con Beta-binomial (estimador de probabilidad) | [0,1] |
| score_volatilidad | 1.4826·MAD de los residuos log después de quitar el calendario | ≥ 0 |
| score_predictibilidad | max(0, 1 − WAPE fuera de muestra de los últimos 60 días) | [0,1] |
| tipo_perfil | reglas por prioridad: ALTO_RIESGO > QUIEBRE_INTRADIA > SOBREDOTADA (por cobertura) > DISPERSION > VOLATIL > ESTABLE; umbrales τ absolutos, calibrados y validados en la Etapa 4 | categoría |

**Hija `PERFIL_RETIRO_SUCURSAL_CORTE`:** un intervalo por tramo existente en la zona de la sucursal, en **hora local efectiva**: apertura (9:00 locales, o 8:00 si es muy transaccional) → cada corte (9/11/13/17 h del centro + huso + desfase de 0.85 h, EV-11). Por ejemplo, 0900-0951. Incluye días observados, promedio, p90 y p95 del retiro del intervalo ($) y participación mediana. *(Actualizado en v1.1.0; antes decía intervalos fijos 0800–0900…, que no es lo que el código hace desde EV-11.)*

## 4. Nivel de confianza (evidencia fuera de muestra, no un score inventado)
| Nivel | Regla |
|---|---|
| ALTA | WAPE de los últimos 60 días ≤ 20% y cobertura de P10–P90 en [72%, 88%] |
| MEDIA | WAPE ≤ 35% y cobertura en [65%, 92%] |
| BAJA | Los demás casos |
| INSUFICIENTE | < 60 días válidos, o cambio de nivel detectado (CUSUM sobre residuos) en los últimos 20 días |

Los umbrales se revisan con el backtest. Cada predicción lleva sus factores: efectos aplicados con su magnitud, nivel reciente vs histórico, evento emergente y banderas de datos.

## 5. Persistencia
- `id_politica_run = PRS-{fecha_corte}-{version}-{hash8(config + huella de insumos)}`. Es determinista e idempotente.
- Escritura transaccional. Antes de escribir se valida el contrato (todos los CHECK del DDL).
- Destino desde el 2026-11-20 (autorizado): `DATABASES_LEGACY/bancoDBPlanificacion_minimo.db` más las tablas complementarias `PERFIL_RUN`, `PREDICCION_RETIRO_DIARIA` y `PERFIL_EXCLUSION`. Volcado a Postgres `planeacion`. Las pruebas usan `--db`.
- Desde v1.1.0, el resultado se publica solo si dos cómputos independientes coinciden bit a bit (EV-16). Detalle operativo en `docs/05_operacion.md`.
