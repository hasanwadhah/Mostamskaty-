@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python -m venv .venv
) else (
    py -3.12 -m venv .venv
)
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation completed. Open run_windows.bat to start Mostamskaty.
pause
exit /b 0
:failed
echo Installation failed. Install Python 3.12 x64, check your Internet connection and try again.
pause
exit /b 1
