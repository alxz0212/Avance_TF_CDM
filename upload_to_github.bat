@echo off
:: Script para subir cambios a GitHub automáticamente
:: Autor: Tu Asistente de IA

echo ==========================================
echo    SUBIENDO CAMBIOS A GITHUB
echo ==========================================
echo.

:: 1. Mostrar estado
echo [1/4] Verificando cambios pendientes...
git status
echo.

:: 2. Pedir mensaje
set /p msg=">> Escribe que cambios hiciste (ej: 'Nuevos graficos'): "

:: 3. Agregar todo
echo.
echo [2/4] Agregando archivos al area de preparacion...
git add .

:: 4. Commit
echo.
echo [3/4] Guardando cambios en local...
git commit -m "%msg%"

:: 5. Push
echo.
echo [4/4] Subiendo a GitHub...
git push origin main

echo.
echo ==========================================
echo    ¡LISTO! TUS CAMBIOS ESTAN EN LA NUBE
echo ==========================================
pause
