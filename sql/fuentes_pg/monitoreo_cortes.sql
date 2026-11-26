-- Misma salida que sql/monitoreo_cortes.sql.
SELECT id_fecha, id_sucursal, hora_corte,
       sdo_inicial, sdo_final, depositos, retiros, dotaciones,
       retiros_clientes, total_trans_operaciones_cliente_cargos AS n_trans, in_service
FROM saldos_monitoreo.monitoreo_corte
WHERE id_fecha BETWEEN %(desde)s AND %(hasta)s
  AND hora_corte IN (900, 1100, 1300, 1700)
ORDER BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
