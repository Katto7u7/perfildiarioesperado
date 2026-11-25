SELECT evento_codigo, id_sucursal, id_fecha_solicitud, id_fecha_evento
FROM dboperativo_u.programas_emergentes
ORDER BY id_sucursal, id_fecha_solicitud, evento_codigo, id_fecha_evento
