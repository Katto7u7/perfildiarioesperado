-- Calendario de pagos por letra. Publicado por adelantado (supuesto documentado; EV verifica).
-- Solo calendario: los montos de DISPERSION son proyectados y no se leen.
SELECT id_fecha, id_programa, letra, is_second_letterday
FROM DISPERSION
