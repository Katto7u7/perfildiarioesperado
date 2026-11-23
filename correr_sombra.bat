@echo off
REM Corrida nocturna (perfil + prediccion del siguiente dia operativo). Programar a las 22:00, lunes a viernes.
REM ESCRIBE en la base destino de config\config.yaml (DATABASES_LEGACY\bancoDBPlanificacion_minimo.db, con respaldo previo)
REM y, si destino.postgres.habilitado = true, vuelca a Postgres. Las bases fuente se abren solo lectura.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
echo ===== %DATE% %TIME% inicio corrida ===== >> out\logs\sombra.log
C:\ProgramData\anaconda3\python.exe -m perfil_retiro.cli run-daily --auto --excel >> out\logs\sombra.log 2>&1
echo ===== %DATE% %TIME% fin corrida, codigo %ERRORLEVEL% (0 OK, 2 abortada a proposito, 1 error) ===== >> out\logs\sombra.log
REM Viernes: reporte semanal de monitoreo (codigo 3 = hay alertas; ver out\monitoreo\)
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek"') do set DIA=%%d
if /i "%DIA%"=="Friday" C:\ProgramData\anaconda3\python.exe -m perfil_retiro.monitoreo >> out\logs\sombra.log 2>&1
