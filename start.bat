@echo off
REM Startup script for Traffic Management System (Windows)

echo.
echo ========================================
echo  Traffic Management System Startup
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed.
    echo Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python 3 is not installed.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Node.js version:
node --version
echo.
echo Python version:
python --version
echo.

REM Start backend
echo Starting Backend Server...
cd backend
pip install -r requirements.txt >nul 2>&1
start "Traffic Backend" python main.py
timeout /t 2 /nobreak

REM Start frontend
echo Starting Frontend Server...
cd ..\frontend
call npm install >nul 2>&1
start "Traffic Frontend" npm run dev
timeout /t 3 /nobreak

echo.
echo ========================================
echo  System Status: RUNNING
echo ========================================
echo.
echo Frontend: http://localhost:5173
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Close this window to stop all servers.
echo.

pause
