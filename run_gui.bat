@echo off
echo Starting AI Agent GUI...
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the GUI application
python gui.py

REM Deactivate virtual environment
deactivate

pause
