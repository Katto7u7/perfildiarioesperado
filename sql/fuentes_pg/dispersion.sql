-- Calendario de pagos por letra. Las filas repetidas se consolidan en sources.agrupar_dispersion.
SELECT id_fecha, id_programa, letra, is_second_letterday::int AS is_second_letterday,
       numero_beneficiarios, monto::double precision AS monto
FROM sucursal.dispersion
ORDER BY 1, 2, 3, 4, 5, 6
