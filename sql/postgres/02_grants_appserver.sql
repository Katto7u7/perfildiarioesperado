-- =====================================================================================================
-- Perfil de retiro por sucursal: PERMISOS para el usuario de la aplicación (appserver)
-- Base: sucursales   Esquema: planeacion       Fecha: 2026-11-20
--
-- Por qué: el script 01_planeacion_perfil_retiro.sql ya se aplicó (las 4 tablas existen, dueño alan.vilchis),
-- pero sus GRANT venían comentados. Verificado hoy con has_table_privilege como appserver:
--   perfil_retiro_sucursal        SELECT/INSERT  -> SÍ
--   perfil_retiro_sucursal_corte  SELECT/INSERT  -> NO
--   perfil_run                    SELECT/INSERT  -> NO
--   perfil_exclusion              SELECT/INSERT  -> NO
--   prediccion_retiro_diaria      SELECT/INSERT  -> NO
-- Sin estos permisos el volcado diario falla con "permission denied for table perfil_run".
--
-- Qué hace: solo otorga permisos. No crea, modifica ni borra tablas ni datos. Idempotente.
-- =====================================================================================================

BEGIN;
GRANT SELECT, INSERT         ON planeacion.perfil_retiro_sucursal_corte TO appserver;
GRANT SELECT, INSERT, UPDATE ON planeacion.perfil_run                   TO appserver;
GRANT SELECT, INSERT         ON planeacion.perfil_exclusion             TO appserver;
GRANT SELECT, INSERT         ON planeacion.prediccion_retiro_diaria     TO appserver;
COMMIT;

-- Verificación (ejecutar después; no modifica nada). Las 4 filas deben salir true / true:
-- SELECT t, has_table_privilege('appserver', 'planeacion.' || t, 'SELECT') AS sel,
--           has_table_privilege('appserver', 'planeacion.' || t, 'INSERT') AS ins
--   FROM unnest(ARRAY['perfil_retiro_sucursal_corte','perfil_run','perfil_exclusion','prediccion_retiro_diaria']) AS t;
