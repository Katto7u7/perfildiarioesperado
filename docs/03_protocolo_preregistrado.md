# Protocolo pre-registrado de validación (se congela ANTES de ver resultados de modelos)

**Fecha de registro:** 2026-11-18. Cualquier cambio posterior se anota en §8 con fecha y motivo. Los resultados del holdout se reportan siempre, sean buenos o malos.

## 1. Pregunta
¿El sistema propuesto (M1) pronostica el retiro del siguiente día operativo, y su riesgo de saldo, mejor que alternativas simples, que el método actual y que Fenix, con intervalos que signifiquen lo que dicen?

## 2. Periodos
| Uso | Fechas objetivo | Regla |
|---|---|---|
| Calentamiento | 2025-06-02 → 2025-11-30 | Solo historia; no se evalúa |
| **Desarrollo** | 2025-12-01 → 2026-09-30 | Selección de modelo, vida media del nivel, umbrales τ, ablaciones |
| **Holdout** | 2026-10-01 → 2026-11-13 | Se evalúa **una sola vez**, con todo congelado (código + configuración + hash) |

## 3. Simulación
- **Origen = cada día operativo D.** Se usan solo datos con `id_fecha ≤ D`. El calendario publicado (FECHA, cierres bancarios, DISPERSION) se permite para D+1.
- Los eventos emergentes cuentan solo si `id_fecha_solicitud ≤ D`.
- **Objetivo** = retiro de MOVIMIENTOS del siguiente día operativo (horizonte de 1 a 4 días naturales).
- Se evalúan solo los días objetivo con estado OK. Los días en cuarentena no cuentan ni a favor ni en contra.

## 4. Candidatos
1. **B0** ingenuo: el mismo día de la semana de la semana anterior.
2. **B1** mediana de los últimos 8 días válidos del mismo contexto (tipo de día AM1 / AM2 / BECAS / SIN × régimen).
3. **B2** media de los últimos 20 días válidos.
4. **NB** método del notebook ("mediana 3 ciclos" con letras), reimplementado tal cual y también corregido.
5. **M1** jerárquico robusto (docs/02 §1).
6. **M2** GBM global por cuantiles (scikit-learn), con las mismas variables más rezagos.
7. **ENS** mediana de M1 y M2, y combinación ponderada por pinball.
8. **Fenix** (`PREDICCION_DS.pred_max`), solo como referencia: no hay prueba de cuándo se generó.

## 5. Métricas
- **Puntuales:** WAPE = Σ|e| / Σy (principal), MAE, RMSE, sesgo = Σe / Σy, error log mediano. Se calculan global, **por sucursal**, por tipo de día, por mes, por tamaño (quintil de retiro medio) y en lunes/post-cierre.
- **Probabilísticas:** pinball en P10/P50/P90/P95, cobertura de P10–P90 y de P95 (marginal y condicional por decil de ancho, tamaño y tipo de día), PIT.
- **Riesgo:** evento = saldo mínimo en cortes del día objetivo < factor × P90 pronosticado. Brier (con descomposición), curva de confiabilidad por deciles, AUC, precisión/recall en el umbral de alerta.
- **Perfil:** kappa de Cohen de `tipo_perfil` entre corridas semanales y mensuales; validez predictiva de cada indicador (AUC o Spearman contra el comportamiento de los siguientes 20 días).

## 6. Criterios de aceptación (todos en desarrollo y confirmados en holdout)
1. **WAPE global de M1 ≤ 0.90 × WAPE de B1** (mejora ≥ 10%), con IC95 por bootstrap de bloques (sucursal × semana) que excluya 0, y Diebold-Mariano con p < 0.05.
2. M1 no peor que B1 en **≥ 70%** de las sucursales (por WAPE de cada sucursal).
3. **|sesgo global| ≤ 3%**, y |sesgo| ≤ 5% en cada tipo de día.
4. **Cobertura de P10–P90 entre 77% y 83%** global, y entre 70% y 90% en ≥ 85% de las sucursales.
5. **P(crítico) calibrada:** error absoluto ≤ 5 pp en cada decil con al menos 200 casos.
6. **Holdout:** WAPE de M1 dentro de +15% (relativo) de su WAPE en desarrollo. Si se degrada más, se reporta como inestabilidad y no se declara productivo.
7. Si M2 o ENS superan a M1 con significancia (DM p < 0.05 y mejora ≥ 3%), se adopta el mejor. Si no, gana **M1 por simplicidad**.
8. Las sucursales que no cumplan el criterio 4 o tengan WAPE > 35% quedan con nivel de confianza **BAJA** y se reportan explícitamente.

## 7. Pruebas de integridad obligatorias
- **Fuga:** al alterar los datos posteriores a D, el pronóstico para D+1 debe quedar idéntico. **Control positivo:** inyectar el valor real de D+1 debe mejorar las métricas de forma drástica (demuestra que la prueba detecta fuga).
- **Reproducibilidad:** la reejecución de un origen produce un resultado idéntico, byte a byte.
- **Sensibilidad:** factor del umbral crítico (0.8 / 1.0 / 1.2), tolerancia de traspaso, inclusión de la cuarentena y apertura (07:30 / 08:00 / 08:30). Las conclusiones de los criterios 1 a 4 no deben invertirse.

## 8. Registro de cambios al protocolo
| Fecha | Cambio | Motivo |
|---|---|---|
| 2026-11-18 | Versión inicial | — |
| 2026-11-18 | M1 pasa a nivel por sucursal × régimen y posición en el mes por régimen | Diagnóstico en desarrollo: en meses IMPAR casi todos los días son de letra AM, así que el régimen y el día AM se confunden y el nivel sale sesgado (sesgo −45% en la 1ª semana de 2025-12). Es consistente con EV-09 |
| 2026-11-25 (1.2.0) | El tamaño del pago y las señales de arrastre pasan de `DISPERSION.monto` al retiro real de MOVIMIENTOS (solo fechas anteriores) | Decisión del usuario: `DISPERSION` es un calendario proyectado que casi nunca se cumple. El holdout ya fue visto: la comparación no es una validación virgen |
| 2026-11-18 | Variables nuevas de "saldo pendiente de dispersión" (AM/BECAS, vida media de 2 y 8 días hábiles) | El retiro de una dispersión continúa días después (arrastre de AM; BECAS desfasado de su calendario) |
| 2026-11-18 | Regresión a nivel fecha acotada (β ≥ 0 en saldos pendientes y días de letra) + ridge 0.01 | Sin la restricción aparecía β < 0 en el saldo AM (físicamente imposible) por confusión con el nivel |
| 2026-11-18 | Desviación propia por combinación de letras, contraída (n/(n+2)) | El método del notebook demostró que la respuesta a cada letra es propia de cada sucursal |
| 2026-11-18 | Vida media del nivel = 4 días hábiles | Selección en la muestra de desarrollo (1 de cada 7 orígenes): h = 3/4/7/10/20/40 → WAPE 0.377/0.377/0.381/0.385/0.393/0.397 |
| 2026-11-18 | Candidato nuevo **ENS_AM**: días AM = promedio(M1, NB); días no AM = M1 | Combinación simple e interpretable. Muestra de desarrollo: WAPE 0.352 contra 0.377 (M1) y 0.410 (NB). Se evalúa con el mismo protocolo, sin privilegios |
| 2026-11-18 | Media por smearing de Duan; intervalos por conformal causal sobre errores fuera de muestra del predictor final | La media por rejilla de cuantiles sobreestimaba (+8%); los residuos de nivel no son completamente fuera de muestra (cobertura 0.75) |
| 2026-11-18 | Candidato **ENS_ALL**: días AM = promedio(M1, NB, M2); otros días = promedio(M1, M2) | Extiende ENS_AM y la combinación pre-registrada M1+M2 |
| 2026-11-18 | Ganador en desarrollo: ENS_ALL (WAPE 0.3188), equivalente a M2 (0.3194; DM p = 0.077). Se adopta ENS_ALL | Regla §6.7: supera a M1 con significancia (13%, p ≈ 0). Ante la equivalencia con M2 se prefiere la combinación: conserva el componente interpretable (factores de M1) y diversifica el riesgo de que un solo modelo falle |
| 2026-11-18 | Media = estimador de razón Σreal/Σpronóstico (40 días, por estrato, con razón propia contraída). Conformal: 120 errores por sucursal, n0 = 10 | En desarrollo, el smearing daba sesgo +7.4% (falla del criterio 3) y la configuración inicial daba 77% de sucursales con cobertura en [70, 90] (falla del 4b). Con el ajuste: sesgo +0.01%, 91.9% de sucursales. Ajuste hecho solo con datos de desarrollo, antes del holdout |
| 2026-11-18 | **Descartado:** modelo top-down (participación de la sucursal × total nacional por programa) | Aun con el total nacional real conocido (oráculo), WAPE = 0.468, peor que M1 |
| 2026-11-18 (post-holdout) | Holdout evaluado una sola vez (EV-13): cumple 1, 2, 4 y 6; **falla 3 (sesgo +6.0%), 4b (75.8% de sucursales) y 5 (5.2 pp)**. Se probó una razón por régimen: empeoró (+8.2%) y **no se adopta** | Reporte íntegro; el sistema no se declara listo para operación plena |
| 2026-11-18 (post-holdout) | **Definición de riesgo (decisión del usuario):** crítico = saldo de CIERRE < P90 de la siguiente jornada; quiebre intradía = saldo en un corte < P90 de la demanda RESTANTE (17:00: saldo < 0). Se recalibran los umbrales de tipo_perfil (EV-14: ALTO_RIESGO 0.10, QUIEBRE 0.07) | La definición anterior aplicaba el umbral del día completo dentro del día (P(crítico) = 1 trivial). La evaluación del holdout con la nueva definición (EV-15) es POST HOC |
| 2026-11-18 | Operación en sombra de 4–6 semanas, con los criterios 3, 4b y 5 como condiciones de salida medidas prospectivamente (`perfil_retiro/monitoreo.py`) | Validación confirmatoria con datos que todavía no existen |
