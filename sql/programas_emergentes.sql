-- Eventos LEEN/LCEN. Se lee completa; el pipeline filtra id_fecha_solicitud <= fecha de corte.
SELECT evento_codigo, id_sucursal, id_fecha_solicitud, id_fecha_evento
FROM PROGRAMAS_EMERGENTES
ORDER BY id_sucursal, id_fecha_solicitud, evento_codigo
