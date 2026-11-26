-- Misma salida que sql/movimientos_diario.sql. SUM en numeric (exacta, no depende del orden) y luego a double.
-- ORDER BY obligatorio: la huella de insumos depende del orden de las filas.
SELECT id_fecha, id_sucursal, id_programa,
       SUM(monto)::double precision AS monto,
       SUM(nu_movimientos)          AS nu_mov,
       COUNT(*)                     AS n_filas,
       COUNT(DISTINCT monto)        AS n_montos_distintos
FROM sucursal.movimientos
WHERE id_fecha BETWEEN %(desde)s AND %(hasta)s
GROUP BY id_fecha, id_sucursal, id_programa
ORDER BY id_fecha, id_sucursal, id_programa
