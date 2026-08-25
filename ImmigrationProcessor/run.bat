@echo off
REM ---------------------------------------------------------------------------
REM Immigration Document Processor - one click launcher (Windows)
REM
REM   Double click this file, or run it from cmd:  run.bat
REM   A different port:  set PORT=9000 && run.bat
REM
REM Creates a local virtual environment, installs the dependencies once,
REM generates demo PDFs on first start and opens the app in the browser.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%PORT%"=="" set PORT=8501
set VENV_DIR=.venv
set STAMP=%VENV_DIR%\.requirements.installed

REM 1) Find Python ------------------------------------------------------------
set PYTHON_BIN=
where py >nul 2>&1 && set PYTHON_BIN=py -3
if "%PYTHON_BIN%"=="" (
    where python >nul 2>&1 && set PYTHON_BIN=python
)
if "%PYTHON_BIN%"=="" (
    echo ERROR: Python 3 was not found. Install it from https://www.python.org/downloads/
    echo Remember to tick "Add python.exe to PATH" in the installer.
    pause
    exit /b 1
)

REM 2) Virtual environment ----------------------------------------------------
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR% ...
    %PYTHON_BIN% -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo ERROR: could not create the virtual environment. & pause & exit /b 1 )
)
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

REM 3) Dependencies -----------------------------------------------------------
if not exist "%STAMP%" (
    echo Installing dependencies ^(this takes a minute on the first run^) ...
    "%VENV_PYTHON%" -m pip install --upgrade pip --quiet
    "%VENV_PYTHON%" -m pip install -r requirements.txt --quiet
    if errorlevel 1 ( echo ERROR: dependency installation failed. & pause & exit /b 1 )
    echo installed> "%STAMP%"
)

REM 4) Demo PDFs --------------------------------------------------------------
if not exist "data\incoming\samples" (
    "%VENV_PYTHON%" create_test_pdfs.py
)

REM 5) Optional local OCR engine ----------------------------------------------
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo NOTE: tesseract is not installed - scanned PDFs cannot be read locally.
    echo       Install it with:  winget install -e --id UB-Mannheim.TesseractOCR
    echo       ^(Digital PDFs work regardless.^)
)

REM 6) Start ------------------------------------------------------------------
echo.
echo Starting Immigration Document Processor on http://localhost:%PORT%
echo Close this window or press Ctrl+C to stop.
echo.
REM Open the browser a few seconds after the server started. Streamlit itself
REM runs headless, which also skips its first-run e-mail prompt.
REM NO_BROWSER=1 is set by the Claude Code desktop preview, which opens the
REM page in its own Browser pane.
if "%NO_BROWSER%"=="" start "" /b cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:%PORT%"
"%VENV_PYTHON%" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false
if "%NO_BROWSER%"=="" pause
