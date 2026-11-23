-- Calendario de pagos por letra. Publicado por adelantado (supuesto documentado; EV verifica).
SELECT id_fecha, id_programa, letra, is_pago_doble, is_bussnes, is_second_letterday,
       numero_beneficiarios, CAST(monto AS REAL) AS monto, peso_final
FROM DISPERSION
