@echo off
setlocal
set "PYTHONUTF8=1"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PLATE_PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\work\plate_env\Scripts\python.exe" (
    set "PLATE_PYTHON=%~dp0..\..\work\plate_env\Scripts\python.exe"
) else (
    set "PLATE_PYTHON=python"
)
"%PLATE_PYTHON%" "%~dp0run.py"
set "PLATE_STATUS=%ERRORLEVEL%"
if /i not "%~1"=="--no-pause" pause
exit /b %PLATE_STATUS%
