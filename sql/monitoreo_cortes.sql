-- Cortes intradía acumulados (forma intradía y saldos). Solo cortes estándar.
SELECT id_fecha, id_sucursal, hora_corte,
       sdo_inicial, sdo_final, depositos, retiros, dotaciones,
       retiros_clientes, total_trans_operaciones_cliente_cargos AS n_trans, in_service
FROM MONITOREO_CORTE
WHERE id_fecha BETWEEN :desde AND :hasta
  AND hora_corte IN (900, 1100, 1300, 1700)
