SELECT id_fecha, id_sucursal, sdo_inicial, depositos, retiros, sdo_final,
       dotaciones_recibidas_apunte_manual AS dotaciones,
       total_trans_operaciones_cliente_cargos AS n_trans, retiros_clientes
FROM SALDOS
WHERE id_fecha BETWEEN :desde AND :hasta
