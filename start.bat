@echo off
REM Double-click this file to launch the transcription web app.
REM Starts the server and opens your browser to http://localhost:5000

cd /d "%~dp0"

REM Verify Python is installed and on PATH
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not on PATH.
    echo.
    echo Install Python from https://www.python.org/downloads/
    echo On the first installer screen, check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

REM Quietly install/refresh dependencies (fast if already installed)
echo Checking Python dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install Python dependencies.
    echo Try running:  pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Verify ffmpeg is on PATH (needed for audio decoding)
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: ffmpeg was not found on PATH.
    echo mp3/m4a/etc. decoding will fail. Install with:
    echo   winget install Gyan.FFmpeg
    echo Then reopen this window.
    echo.
    pause
)

REM Open the browser after a short delay (once the server is up)
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

echo.
echo ==============================================
echo   Transcription server starting...
echo   Open in browser: http://localhost:5000
echo   Close this window or press Ctrl+C to stop.
echo ==============================================
echo.

python app.py

REM If Python exits (crash or Ctrl+C), keep the window open so the user can read errors
echo.
echo Server stopped.
pause
