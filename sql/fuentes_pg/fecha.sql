-- Misma salida que sql/fecha.sql (banderas como entero).
SELECT id_fecha, dia, mes, anio, dia_semana, is_weekend::int AS is_weekend, is_holiday::int AS is_holiday
FROM sucursal.fecha
ORDER BY id_fecha
