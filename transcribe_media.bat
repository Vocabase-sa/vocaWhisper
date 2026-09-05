@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage:
    echo   transcribe_media.bat "C:\chemin\vers\video.mp4"
    echo   transcribe_media.bat "C:\chemin\vers\video.mp4" "C:\chemin\vers\sortie.txt"
    exit /b 1
)

if "%~2"=="" (
    "%~dp0venv\Scripts\python.exe" "%~dp0batch\transcribe_media_to_txt.py" "%~1"
) else (
    "%~dp0venv\Scripts\python.exe" "%~dp0batch\transcribe_media_to_txt.py" "%~1" --output "%~2"
)
