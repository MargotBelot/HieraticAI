@echo off
REM HieraticAI Validation Interface Launcher for Windows
REM Portable script that works for any user

echo Starting HieraticAI Validation Interface...

REM Get the directory where this script is located
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if exist "hieratic_env\" (
    echo Activating virtual environment...
    call "hieratic_env\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found at hieratic_env\
    echo Continuing without virtual environment activation...
)

REM Check if streamlit is available
where streamlit >nul 2>nul
if errorlevel 1 (
    echo Error: streamlit is not installed
    echo Please install it: pip install streamlit
    pause
    exit /b 1
)

REM Check if the validation script exists
if not exist "tools\validation\prediction_validator.py" (
    echo Error: Validation script not found at tools\validation\prediction_validator.py
    pause
    exit /b 1
)

REM Launch Streamlit
echo Launching Streamlit interface...
echo The interface will open in your browser automatically.
echo Press Ctrl+C to stop the server.
echo.

REM Start Streamlit
streamlit run tools\validation\prediction_validator.py

pause
