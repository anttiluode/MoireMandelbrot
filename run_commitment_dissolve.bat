@echo off
cd /d "%~dp0"
python commitment_dissolve_app.py
if errorlevel 1 pause
