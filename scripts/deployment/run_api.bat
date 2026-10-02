@echo off
echo ====================================================
echo Starting Behavior Detection API Server (FastAPI)
echo ====================================================
cd /d "%~dp0..\.."
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
pause