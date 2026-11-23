#!/bin/bash
# Backtest por meses en procesos independientes con reintento. Uso: bash backtest_por_meses.sh ETIQUETA MES1 MES2 ...
cd "$(dirname "$0")"
tag=$1; shift
for ym in "$@"; do
  d="${ym}01"; h="${ym}31"
  for intento in 1 2 3; do
    PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore C:/ProgramData/anaconda3/python.exe -u correr_backtest.py $d $h ${tag}_$ym > ../out/backtest/${tag}_$ym.log 2>&1 && break
    echo "$ym intento $intento falló: $(tail -1 ../out/backtest/${tag}_$ym.log)" >> ../out/backtest/${tag}_fallas.log
  done
done
