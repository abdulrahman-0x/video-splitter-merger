@echo off
setlocal
cd /d "%~dp0"
title Video Splitter and Merger

REM 1. Check for Python
where python >nul 2>&1
if errorlevel 1 goto NoPython

REM 2. Check if virtual environment exists
if exist "%~dp0.venv\Scripts\python.exe" goto CheckDeps

echo [1/2] Creating virtual environment...
python -m venv "%~dp0.venv"
if errorlevel 1 goto VenvError

echo [2/2] Installing dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto PipError
goto LaunchApp

:CheckDeps
"%~dp0.venv\Scripts\python.exe" -c "import customtkinter" >nul 2>&1
if not errorlevel 1 goto LaunchApp

echo Installing dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto PipError

:LaunchApp
start "" "%~dp0.venv\Scripts\pythonw.exe" -m src.main
exit /b 0

:NoPython
echo.
echo [ERROR] Python was not found in PATH.
echo Please install Python 3.10 or newer from:
echo https://www.python.org/downloads/
echo Make sure to check 'Add Python to PATH' during installation.
echo.
pause
exit /b 1

:VenvError
echo.
echo [ERROR] Failed to create virtual environment.
echo.
pause
exit /b 1

:PipError
echo.
echo [ERROR] Failed to install required packages.
echo.
pause
exit /b 1
