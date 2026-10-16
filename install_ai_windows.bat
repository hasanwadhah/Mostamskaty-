@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run install_windows.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto failed
echo Local AI installation completed. Future detection works offline.
pause
exit /b 0
:failed
echo AI setup failed. The application can still use OpenCV edge detection.
pause
exit /b 1
