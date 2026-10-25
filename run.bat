@echo off
chcp 65001 >nul
title Video Splitter & Merger

cd /d "%~dp0"

:: Check for Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [خطأ] لم يتم العثور على بايثون (Python) في النظام.
    echo [Error] Python was not found on your system.
    echo.
    echo يرجى تثبيت Python 3.10 أو أحدث وتفعيل خيار "Add Python to PATH" أثناء التثبيت.
    echo Please install Python 3.10 or newer and check "Add Python to PATH".
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Create Virtual Environment if not exists
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] جاري إنشاء البيئة الافتراضية (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [خطأ] فشل إنشاء البيئة الافتراضية.
        pause
        exit /b 1
    )
    echo [2/2] جاري تثبيت الحزم المطلوبة...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [خطأ] فشل تثبيت الحزم المطلوبة.
        pause
        exit /b 1
    )
    echo تم التثبيت بنجاح.
) else (
    :: Quick verification of dependencies
    ".venv\Scripts\python.exe" -c "import customtkinter" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo جاري استكمال تثبيت الحزم المطلوبة...
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    )
)

:: Launch application with pythonw (no black console window)
start "" ".venv\Scripts\pythonw.exe" -m src.main
exit /b 0
