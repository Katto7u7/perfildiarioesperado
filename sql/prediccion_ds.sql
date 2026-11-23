-- Referencia externa (Fenix). Sin timestamp de generación: NUNCA se usa como insumo.
SELECT id_sucursal, fecha AS id_fecha, id_modelo, pred_max FROM PREDICCION_DS
WHERE fecha BETWEEN :desde AND :hasta
