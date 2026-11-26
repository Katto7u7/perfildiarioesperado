-- Calendario de pagos por letra. Las filas repetidas se consolidan en sources.agrupar_dispersion.
-- Solo calendario: los montos de DISPERSION son proyectados y no se leen.
SELECT id_fecha, id_programa, letra, is_second_letterday::int AS is_second_letterday
FROM sucursal.dispersion
ORDER BY 1, 2, 3, 4
