@echo off
echo ====================================================
echo Starting Abnormal Behavior Detection System...
echo ====================================================
echo.

REM Start API backend in new window
start "API Server" cmd /k "cd /d %~dp0..\.. && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait 3 seconds for API to initialize
timeout /t 3 /nobreak >nul

REM Start React frontend in new window
start "Web Interface" cmd /k "cd /d %~dp0..\..\behavior-detection-web && npm run dev"

echo.
echo System is starting...
echo API Backend:  http://localhost:8000
echo Web Frontend: http://localhost:3000
echo.
pause