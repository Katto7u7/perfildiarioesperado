# Diccionario de indicadores — `PERFIL_RETIRO_SUCURSAL` y tablas complementarias

**Convenciones**
- *Retiro* = retiro de clientes de `bancoDB11.MOVIMIENTOS` (decisión 1, EV-01), sumado por sucursal-día.
- *Día válido* = día operativo con estado OK (EV-02). No se imputa nada.
- *Ventana del perfil* = últimos **126 días válidos del régimen vigente**: IMPAR = meses de AM, PAR = meses de BECAS. El régimen vigente es el del día objetivo (EV-09).
- *Umbrales del perfil* (60 días válidos del mismo régimen): para el **cierre** se usa el P90 que incluye el día d (se conoce al cierre de d y estima la jornada siguiente); para el **quiebre intradía** se usa el P90 de los días **anteriores** a d (se conoce antes de abrir). Ambos son causales.
- *Hora local*: los cortes 9/11/13/17 están en hora del centro. Se convierten con el huso de la sucursal y el desfase efectivo δ = 0.85 h (EV-11).
- Implementación: `perfil_retiro/perfil.py`. Validación: `perfil_retiro/contrato.py`.

| Columna | Fórmula | Unidad | Significado operativo | Notas / evidencia |
|---|---|---|---|---|
| id_politica_run | `PRS-{fecha_corte}-{versión}-{hash8(config + huella de insumos)}` | texto | Identifica la corrida; los mismos insumos producen el mismo id | Idempotente (`persistencia.py`) |
| id_sucursal | — | — | Sucursal | Solo las que cumplen el contrato; el resto va a `PERFIL_EXCLUSION` |
| dias_observados | nº de días válidos en la ventana | días | Cuánta evidencia respalda el perfil | Mínimo 60; si no se alcanza, se excluye |
| retiro_promedio_diario | media del retiro diario | $ | Demanda típica del régimen | EV-09 |
| retiro_p90_diario | cuantil 0.90 (lineal) | $ | Día alto "normal": 1 de cada 10 días lo supera | CHECK ≥ promedio |
| retiro_p95_diario | cuantil 0.95 | $ | Día muy alto | CHECK ≥ p90 |
| retiro_max_diario | máximo | $ | Peor día observado en la ventana | CHECK ≥ p95 |
| volatilidad_retiro_diario | desviación estándar (ddof = 1) | $ | Dispersión absoluta del retiro | Incluye la variación predecible por calendario (ver score_volatilidad) |
| dias_cobertura_promedio | media de min(saldo_cierre(d) / P50 causal, 30) | días | Cuántos días típicos cubre el saldo de cierre | Tope de 30 contra la explosión cuando la demanda es ~0 |
| dias_cobertura_p90 | media de min(saldo_cierre(d) / P90 causal, 30) | días | Cobertura ante una demanda alta | **El nombre no es el percentil 90 de la cobertura**: es la cobertura ante demanda P90 |
| saldo_promedio_cierre | media de `sdo_final`@17:00 en días con saldo válido | $ | Efectivo con el que la sucursal suele cerrar | Excluye cierres negativos (error de dato) |
| pct_dias_saldo_critico | % de días con **saldo de cierre** < P90 de la siguiente jornada (estimado al cierre) | [0,1] | Frecuencia con la que la sucursal cierra con menos de un día de demanda alta (decisión del usuario) | Saldo real (incluye dotaciones del día) |
| velocidad_consumo_promedio | media entre días del ritmo de la jornada = retiro del día ÷ horas de operación efectivas | $/h | Ritmo típico de salida de efectivo | Hora local, con apertura y δ (EV-11); redefinida el 2026-11-20 (antes: un solo pool de intervalos) |
| velocidad_consumo_p90 | p90 entre días de la velocidad del tramo pico del día | $/h | Ritmo en el tramo más intenso de un día alto | CHECK ≥ promedio por construcción |
| aceleracion_abs_p90 | p90 de \|Δvelocidad\| / Δt entre puntos medios de intervalos consecutivos | $/h² | Qué tan bruscos son los cambios de ritmo | Indicador descriptivo; su valor predictivo se reporta en EV-14 |
| indice_agresividad_consumo | mediana de [fracción acumulada al corte de las 11 ÷ fracción de la jornada transcurrida a ese corte] | razón | > 1: el retiro se concentra al inicio de la jornada | Mediana global ≈ 1.6: el arranque siempre es más intenso |
| indice_concentracion_intradia | mediana de Σ pᵢ²/tᵢ − 1 (pᵢ = participación del intervalo, tᵢ = fracción de tiempo) | ≥ 0 | 0 = retiro uniforme en el tiempo; mayor = concentrado en pocos tramos | Divergencia χ² respecto a un ritmo uniforme |
| hora_pico_consumo | hora local de inicio del intervalo con mayor velocidad mediana | HHMM | Cuándo empieza el tramo de mayor presión | Depende de la apertura declarada/inferida (EV-05/EV-11) |
| intervalo_pico | "HHMM-HHMM" local de ese intervalo | texto | Tramo de mayor presión | Por ejemplo, 0900-0951 = desde la apertura hasta el primer corte efectivo |
| participacion_intervalo_pico | mediana de la fracción del retiro del día que ocurre en ese intervalo | [0,1] | Qué tanto pesa el tramo pico | |
| pct_dias_consumo_extremo | % de días con residuo log > 3·MAD respecto a la mediana de su tipo de día (AM1/AM2/BE1/BE2/SIN) | [0,1] | Frecuencia de días anormalmente altos **para su contexto** | No castiga los días de letra esperados |
| score_riesgo_desabasto | (k + 20·m) / (n + 20); k = días en riesgo (cierre crítico **o** quiebre intradía: saldo en un corte < P90 de la demanda restante del día, o < 0 a las 17:00), n = días con saldo, m = tasa global | [0,1] | Probabilidad estimada de que un día sea de riesgo | Beta-binomial: con poca evidencia se acerca a la tasa global |
| score_volatilidad | 1.4826·MAD de los residuos log respecto al tipo de día | log ≈ relativo | Variabilidad **que el calendario no explica** | Es el componente impredecible |
| score_predictibilidad | max(0, 1 − WAPE fuera de muestra de los últimos 60 días del modelo productivo) | [0,1] | Qué tan bien se ha pronosticado esta sucursal **en la realidad** | Antes de la primera corrida productiva sale del backtest; no es un puntaje inventado |
| tipo_perfil | ALTO_RIESGO (score ≥ 0.10) > QUIEBRE_INTRADIA (quiebre ≥ 7% de días) > SOBREDOTADA (cobertura ≥ 20 días y riesgo < 0.05) > DISPERSION (intensidad AM ≥ 10) > VOLATIL (score_volatilidad ≥ 0.9) > ESTABLE | categoría | Clasificación operativa; **se lee junto con su régimen** | EV-14: lift por categoría, kappa 0.79–0.86 dentro del régimen |
| created_at_utc | timestamp UTC de la corrida | — | Auditoría | |

## `PERFIL_RETIRO_SUCURSAL_CORTE` (una fila por intervalo existente en la zona de la sucursal)
| Columna | Fórmula |
|---|---|
| hora_inicio / hora_fin | límites del intervalo en hora local efectiva: apertura → primer corte con operación → siguientes cortes |
| dias_observados_intervalo | días con forma intradía válida (4 cortes, monótona, sin traspaso) |
| retiro_promedio / p90 / p95_intervalo | fracción del intervalo × retiro de MOVIMIENTOS del día; media, p90 y p95 |
| participacion_dia | mediana de la fracción del retiro del día en el intervalo |

## `PREDICCION_RETIRO_DIARIA` (tabla complementaria propuesta, por día objetivo)
| Columna | Significado |
|---|---|
| retiro_esperado | media pronosticada (estimador de razón Σreal/Σpronóstico de los últimos 40 días): útil para sumar entre sucursales. **En el holdout tuvo sesgo +6%; se vigila en la sombra** |
| retiro_p10 / p50 / p90 / p95 | p50 = pronóstico puntual de ENS_ALL (M1 + M2, más el método de letras en días AM); demás cuantiles por conformal causal sobre errores fuera de muestra |
| prob_retiro_extraordinario | P(retiro > p95 del perfil); vacío si la sucursal no tiene perfil |
| saldo_inicial | saldo de cierre del día de corte (17:00) |
| umbral_critico | 1.0 × retiro_p90 pronosticado |
| prob_saldo_critico | P(saldo de cierre < umbral) = P(S0 + depósitos netos − retiro del día < P90), sin dotación adicional. Todas las probabilidades de saldo quedan **vacías** si no hay saldo de cierre del día de corte |
| prob_quiebre_intradia, prob_quiebre_{corte} | P(saldo en el corte < P90 × fracción del día que falta); a las 17:00: P(saldo < 0). Sin dotación adicional |
| corte_presion | primer corte con P(quiebre) ≥ 0.5 (0 = ninguno) |
| frac_900/1100/1300 | fracción acumulada esperada del retiro por corte (sucursal × tipo de día) |
| dias_cobertura_p50 / p90 | saldo inicial / retiro_p50 o retiro_p90 publicados |
| nivel_confianza | ALTA / MEDIA / BAJA / INSUFICIENTE según WAPE y cobertura fuera de muestra de los últimos 60 días |
| factores | JSON con las principales contribuciones (%) al pronóstico (día de letra, retiro reciente, retiro de ciclos previos, día de la semana, letras propias…) |

## `PERFIL_RUN` (bitácora de corridas)
| Columna | Significado |
|---|---|
| id_politica_run, fecha_corte, fecha_objetivo, regimen | Identificación de la corrida |
| version_modelo, config_hash, huella_insumos, parametros | Reproducibilidad: con estos datos la corrida se puede reconstruir |
| estado | OK / OK_CON_RECHAZOS (algún perfil o intervalo no pasó el contrato; detalle en PERFIL_EXCLUSION) / FALLIDA (reservado) |
| n_perfiles, n_cortes, n_exclusiones, n_predicciones, duracion_seg | Resultado |
| mensaje | Rechazos de contrato, avisos (p. ej. DISPERSION aún no cargada para el día objetivo) y resultado del **cómputo verificado** ("2 de 2 coincidentes") |
| vigente | 1 = corrida vigente para su fecha de corte (si los insumos cambian y se recalcula, la anterior queda en 0) |
| modo | SOMBRA / HISTORICO / PRODUCCION / PRUEBA |
| pg_sincronizado | 1 = ya volcada a Postgres |

## `PERFIL_EXCLUSION` (una fila por sucursal y motivo, por corrida)
| motivo | Significado | ¿Se publica el perfil? |
|---|---|---|
| DIAS_VALIDOS_INSUFICIENTES | menos de 60 días válidos del régimen en la ventana | No |
| SIN_DEMANDA | retiro 0 en todos los días válidos de la ventana | No |
| CONTRATO | el perfil violaría algún CHECK/NOT NULL del DDL (el detalle dice cuál); no se corrige ni se fuerza | No |
| CONTRATO_INTERVALO | uno o más intervalos de `_CORTE` violarían el DDL (detalle: intervalo y regla) | Sí, sin esos intervalos |
| FK_SUCURSAL | la sucursal no existe en el catálogo `SUCURSAL` del destino (el catálogo no se modifica) | No |
