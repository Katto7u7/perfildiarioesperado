-- =====================================================================================================
-- Perfil de retiro por sucursal: cambios de estructura en Postgres
-- Base: sucursales   Esquema: planeacion
-- Solicitado por: equipo de Planeación / proyecto "perfil de retiro por sucursal"
-- Fecha: 2026-11-20
--
-- Qué hace este script (todo es idempotente: IF NOT EXISTS, se puede ejecutar más de una vez):
--   1. CREA planeacion.perfil_retiro_sucursal_corte  (tabla hija de planeacion.perfil_retiro_sucursal, que ya existe)
--   2. CREA 3 tablas complementarias del mismo proceso:
--        planeacion.perfil_run                 (bitácora de corridas; una fila por id_politica_run)
--        planeacion.perfil_exclusion           (sucursales sin perfil en una corrida y el motivo)
--        planeacion.prediccion_retiro_diaria   (pronóstico del siguiente día operativo por sucursal)
--   3. (Opcional, comentado) permisos para el usuario de la aplicación.
--
-- Qué NO hace: no modifica ni borra ninguna tabla existente (perfil_retiro_sucursal, sucursal, politica_*, forecast_diario).
-- Estructura idéntica a las tablas equivalentes en SQLite (bancoDBPlanificacion_minimo.db), mismas reglas CHECK.
-- Todo va en una transacción: si algo falla, no queda nada a medias.
-- =====================================================================================================

BEGIN;

-- -----------------------------------------------------------------------------------------------------
-- 1. Intervalos intradía del perfil (hija de perfil_retiro_sucursal)
--    Una fila por (corrida, sucursal, intervalo). Horas en formato HHMM, hora local de la sucursal.
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS planeacion.perfil_retiro_sucursal_corte (
    id_politica_run            text             NOT NULL,
    id_sucursal                integer          NOT NULL,
    hora_inicio                integer          NOT NULL,
    hora_fin                   integer          NOT NULL,
    dias_observados_intervalo  integer          NOT NULL,
    retiro_promedio_intervalo  double precision NOT NULL,
    retiro_p90_intervalo       double precision NOT NULL,
    retiro_p95_intervalo       double precision NOT NULL,
    participacion_dia          double precision NOT NULL,
    created_at_utc             timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT perfil_retiro_sucursal_corte_pkey PRIMARY KEY (id_politica_run, id_sucursal, hora_inicio, hora_fin),
    CONSTRAINT perfil_retiro_sucursal_corte_hora_inicio_check
        CHECK (hora_inicio BETWEEN 0 AND 2359 AND (hora_inicio % 100) BETWEEN 0 AND 59),
    CONSTRAINT perfil_retiro_sucursal_corte_hora_fin_check
        CHECK (hora_fin BETWEEN 0 AND 2359 AND (hora_fin % 100) BETWEEN 0 AND 59 AND hora_fin > hora_inicio),
    CONSTRAINT perfil_retiro_sucursal_corte_dias_check          CHECK (dias_observados_intervalo >= 1),
    CONSTRAINT perfil_retiro_sucursal_corte_promedio_check      CHECK (retiro_promedio_intervalo >= 0),
    CONSTRAINT perfil_retiro_sucursal_corte_p90_check           CHECK (retiro_p90_intervalo >= 0),
    CONSTRAINT perfil_retiro_sucursal_corte_p95_check           CHECK (retiro_p95_intervalo >= 0),
    CONSTRAINT perfil_retiro_sucursal_corte_participacion_check CHECK (participacion_dia >= 0 AND participacion_dia <= 1),
    CONSTRAINT perfil_retiro_sucursal_corte_orden_p95_check     CHECK (retiro_p95_intervalo >= retiro_p90_intervalo),
    CONSTRAINT perfil_retiro_sucursal_corte_orden_p90_check     CHECK (retiro_p90_intervalo >= retiro_promedio_intervalo),
    CONSTRAINT perfil_retiro_sucursal_corte_padre_fkey FOREIGN KEY (id_politica_run, id_sucursal)
        REFERENCES planeacion.perfil_retiro_sucursal (id_politica_run, id_sucursal)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_perfil_retiro_corte_sucursal
    ON planeacion.perfil_retiro_sucursal_corte (id_sucursal, hora_inicio, hora_fin);

COMMENT ON TABLE planeacion.perfil_retiro_sucursal_corte IS
    'Intervalos intradía del perfil de retiro (hija de perfil_retiro_sucursal). Horas HHMM en hora local de la sucursal.';

-- -----------------------------------------------------------------------------------------------------
-- 2a. Bitácora de corridas
--     modo: SOMBRA (validación, aún no oficial) | HISTORICO (reconstrucción) | PRODUCCION | PRUEBA
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS planeacion.perfil_run (
    id_politica_run  text             NOT NULL,
    fecha_corte      integer          NOT NULL,
    fecha_objetivo   integer          NOT NULL,
    regimen          text             NOT NULL,
    version_modelo   text             NOT NULL,
    config_hash      text             NOT NULL,
    huella_insumos   text             NOT NULL,
    parametros       text             NOT NULL,
    estado           text             NOT NULL,
    n_perfiles       integer          NOT NULL,
    n_cortes         integer          NOT NULL,
    n_exclusiones    integer          NOT NULL,
    n_predicciones   integer          NOT NULL,
    duracion_seg     double precision,
    mensaje          text,
    vigente          integer          NOT NULL DEFAULT 1,
    modo             text             NOT NULL DEFAULT 'SOMBRA',
    pg_sincronizado  integer          NOT NULL DEFAULT 0,
    created_at_utc   timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT perfil_run_pkey PRIMARY KEY (id_politica_run),
    CONSTRAINT perfil_run_estado_check  CHECK (estado IN ('OK', 'OK_CON_RECHAZOS', 'FALLIDA')),
    CONSTRAINT perfil_run_modo_check    CHECK (modo IN ('SOMBRA', 'PRODUCCION', 'HISTORICO', 'PRUEBA')),
    CONSTRAINT perfil_run_vigente_check CHECK (vigente IN (0, 1)),
    CONSTRAINT perfil_run_pgsync_check  CHECK (pg_sincronizado IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_perfil_run_fecha ON planeacion.perfil_run (fecha_corte, vigente);

COMMENT ON TABLE planeacion.perfil_run IS
    'Bitácora de corridas del perfil de retiro: fechas, versión, huella de insumos, parámetros, estado y modo.';

-- -----------------------------------------------------------------------------------------------------
-- 2b. Exclusiones por corrida (sucursales sin perfil y motivo)
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS planeacion.perfil_exclusion (
    id_politica_run  text    NOT NULL,
    id_sucursal      integer NOT NULL,
    motivo           text    NOT NULL,
    detalle          text,

    CONSTRAINT perfil_exclusion_pkey PRIMARY KEY (id_politica_run, id_sucursal, motivo)
);

COMMENT ON TABLE planeacion.perfil_exclusion IS
    'Sucursales excluidas del perfil en cada corrida (DIAS_VALIDOS_INSUFICIENTES, SIN_DEMANDA, CONTRATO, FK_SUCURSAL).';

-- -----------------------------------------------------------------------------------------------------
-- 2c. Pronóstico del siguiente día operativo por sucursal
--     Probabilidades suponen que no hay dotación adicional en el día.
-- -----------------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS planeacion.prediccion_retiro_diaria (
    id_politica_run             text             NOT NULL,
    id_sucursal                 integer          NOT NULL,
    fecha_corte                 integer          NOT NULL,
    fecha_objetivo              integer          NOT NULL,
    horizonte_dias              integer          NOT NULL,
    retiro_esperado             double precision NOT NULL,
    retiro_p10                  double precision NOT NULL,
    retiro_p50                  double precision NOT NULL,
    retiro_p90                  double precision NOT NULL,
    retiro_p95                  double precision NOT NULL,
    prob_retiro_extraordinario  double precision,
    saldo_inicial               double precision,
    umbral_critico              double precision NOT NULL,
    prob_saldo_critico          double precision,
    prob_quiebre_intradia       double precision,
    corte_presion               integer,
    prob_quiebre_900            double precision,
    prob_quiebre_1100           double precision,
    prob_quiebre_1300           double precision,
    prob_quiebre_1700           double precision,
    frac_900                    double precision,
    frac_1100                   double precision,
    frac_1300                   double precision,
    dias_cobertura_p50          double precision,
    dias_cobertura_p90          double precision,
    nivel_confianza             text             NOT NULL,
    factores                    text,
    created_at_utc              timestamp without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT prediccion_retiro_diaria_pkey PRIMARY KEY (id_politica_run, id_sucursal, fecha_objetivo),
    CONSTRAINT prediccion_retiro_diaria_cuantiles_check
        CHECK (retiro_p10 <= retiro_p50 AND retiro_p50 <= retiro_p90 AND retiro_p90 <= retiro_p95),
    CONSTRAINT prediccion_retiro_diaria_confianza_check
        CHECK (nivel_confianza IN ('ALTA', 'MEDIA', 'BAJA', 'INSUFICIENTE'))
);

CREATE INDEX IF NOT EXISTS idx_prediccion_retiro_objetivo
    ON planeacion.prediccion_retiro_diaria (fecha_objetivo, id_sucursal);

COMMENT ON TABLE planeacion.prediccion_retiro_diaria IS
    'Pronóstico de retiro del siguiente día operativo por sucursal: esperado, cuantiles, riesgo de saldo y confianza.';

-- -----------------------------------------------------------------------------------------------------
-- 3. Permisos para el usuario de la aplicación (ajustar el rol si corresponde; descomentar si se requiere)
-- -----------------------------------------------------------------------------------------------------
-- GRANT SELECT, INSERT ON planeacion.perfil_retiro_sucursal       TO appserver;
-- GRANT SELECT, INSERT ON planeacion.perfil_retiro_sucursal_corte TO appserver;
-- GRANT SELECT, INSERT, UPDATE ON planeacion.perfil_run           TO appserver;
-- GRANT SELECT, INSERT ON planeacion.perfil_exclusion             TO appserver;
-- GRANT SELECT, INSERT ON planeacion.prediccion_retiro_diaria     TO appserver;

COMMIT;

-- =====================================================================================================
-- Verificación (ejecutar después; no modifica nada)
-- =====================================================================================================
-- SELECT table_name FROM information_schema.tables
--  WHERE table_schema = 'planeacion'
--    AND table_name IN ('perfil_retiro_sucursal', 'perfil_retiro_sucursal_corte', 'perfil_run',
--                       'perfil_exclusion', 'prediccion_retiro_diaria')
--  ORDER BY 1;                                   -- deben salir las 5

-- =====================================================================================================
-- Reversa (solo si se requiere deshacer; borra las 4 tablas nuevas y sus datos)
-- =====================================================================================================
-- BEGIN;
-- DROP TABLE IF EXISTS planeacion.perfil_retiro_sucursal_corte;
-- DROP TABLE IF EXISTS planeacion.prediccion_retiro_diaria;
-- DROP TABLE IF EXISTS planeacion.perfil_exclusion;
-- DROP TABLE IF EXISTS planeacion.perfil_run;
-- COMMIT;
