@echo off
REM Double-click this file to start the Tomato Leaf Screening app on Windows.
REM First run: creates a private Python environment and installs the libraries
REM (about 5-10 minutes, needs internet). Later runs start in a few seconds.
cd /d "%~dp0"

set PY=
python --version >nul 2>nul && set PY=python
if not defined PY py -3 --version >nul 2>nul && set PY=py -3
if not defined PY (
  echo Python was not found.
  echo Install Python 3.10 - 3.13 from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during installation, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment in .venv ...
  %PY% -m venv .venv || goto :fail
)

".venv\Scripts\python.exe" -c "import tensorflow, gradio, sklearn, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo Installing libraries - first time only, please wait ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo.
echo Starting the app. Your browser will open at http://127.0.0.1:7860
echo Close this window to stop the app.
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
