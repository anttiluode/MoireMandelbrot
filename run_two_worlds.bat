@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py two_worlds_app.py
) else (
    python two_worlds_app.py
)

if errorlevel 1 (
    echo.
    echo If dependencies are missing, run:
    echo     pip install -r requirements.txt
    echo.
    pause
)
