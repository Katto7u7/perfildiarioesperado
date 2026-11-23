SELECT id_fecha, id_sucursal, monto FROM INGENIERIA_VARIABLES
WHERE id_fecha BETWEEN :desde AND :hasta
