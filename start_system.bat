@echo off
echo ====================================================
echo Starting Abnormal Behavior Detection System...
echo ====================================================
echo.

REM Start API backend in new window
start "API Server" cmd /k "python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait 3 seconds for API
timeout /t 3 /nobreak >nul

REM Start React Vite Web Interface in new window
start "Web Interface" cmd /k "cd behavior-detection-web && npm run dev"

echo.
echo System started successfully!
echo API Documentation: http://localhost:8000/docs
echo Web Interface:     http://localhost:3000
echo.
pause
