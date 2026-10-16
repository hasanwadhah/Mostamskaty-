@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run install_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" iraqi_id_printer.py
if errorlevel 1 pause
