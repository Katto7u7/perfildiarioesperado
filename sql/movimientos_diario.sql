-- Demanda: retiro de clientes por (fecha, sucursal, programa). Fuente de verdad (decisión 1).
-- n_filas > 1 indica llaves duplicadas en origen.
SELECT id_fecha, id_sucursal, id_programa,
       SUM(CAST(monto AS REAL)) AS monto,
       SUM(nu_movimientos)      AS nu_mov,
       COUNT(*)                 AS n_filas,
       COUNT(DISTINCT CAST(monto AS REAL)) AS n_montos_distintos
FROM MOVIMIENTOS
WHERE id_fecha BETWEEN :desde AND :hasta
GROUP BY id_fecha, id_sucursal, id_programa
