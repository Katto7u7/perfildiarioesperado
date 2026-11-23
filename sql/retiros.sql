SELECT id_fecha, id_sucursal, monto_total, total_movimientos, registros, fuente
FROM RETIROS
WHERE id_fecha BETWEEN :desde AND :hasta
