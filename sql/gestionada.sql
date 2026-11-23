SELECT id_sucursal, fecha, fecha_solicitud, monto, monto_asignado, fecha_entrega, hora_entrega, estatus
FROM GESTIONADA
WHERE fecha BETWEEN :desde AND :hasta
