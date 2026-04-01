@echo off
chcp 65001 >nul 2>&1

REM Se placer dans le dossier du script
cd /d "%~dp0"

echo ============================================================
echo   VocaWhisper - Verification de Python
echo ============================================================
echo.

REM --- Methode 1 : "python" dans le PATH ---
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :FOUND
)

REM --- Methode 2 : Python Launcher "py -3" ---
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :FOUND
)

REM --- Methode 3 : chemins courants ---
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :FOUND
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :FOUND
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :FOUND
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :FOUND
)
if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    goto :FOUND
)

REM --- Python non trouve ---
echo [!] Python n'est pas installe sur ce PC.
echo.
echo     Installez Python 3.12 depuis https://www.python.org/downloads/
echo     IMPORTANT : Cochez "Add Python to PATH" pendant l'installation !
echo     Puis relancez install_windows.bat
echo.
pause
exit /b 1

:FOUND
echo [OK] Python detecte.
%PYTHON_CMD% --version
echo.

:LAUNCH_INSTALLER
echo [*] Lancement de l'installeur graphique...
echo.
%PYTHON_CMD% "%~dp0installer.py"
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installeur a rencontre un probleme.
    pause
)
exit /b 0
