@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM Se placer dans le dossier du script
cd /d "%~dp0"

echo ============================================================
echo   VocaWhisper - Verification de Python
echo ============================================================
echo.

REM --- Methode 1 : "python" dans le PATH ---
python --version >nul 2>&1
if not errorlevel 1 (
    echo [OK] Python detecte :
    python --version
    set "PYTHON_CMD=python"
    goto :LAUNCH_INSTALLER
)

REM --- Methode 2 : Python Launcher "py" (installe avec Python.org) ---
py -3 --version >nul 2>&1
if not errorlevel 1 (
    echo [OK] Python detecte via py launcher :
    py -3 --version
    set "PYTHON_CMD=py -3"
    goto :LAUNCH_INSTALLER
)

REM --- Methode 3 : chercher dans les chemins courants ---
for %%V in (12 11 10) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python3%%V\python.exe" (
        echo [OK] Python 3.%%V trouve dans %LOCALAPPDATA%\Programs\Python\Python3%%V
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python3%%V\python.exe"
        goto :LAUNCH_INSTALLER
    )
    if exist "C:\Python3%%V\python.exe" (
        echo [OK] Python 3.%%V trouve dans C:\Python3%%V
        set "PYTHON_CMD=C:\Python3%%V\python.exe"
        goto :LAUNCH_INSTALLER
    )
)

echo [!] Python n'est pas installe sur ce PC.
echo.

REM --- Methode 4 : Installation automatique via winget ---
winget --version >nul 2>&1
if not errorlevel 1 (
    echo [*] Installation automatique de Python 3.12 via winget...
    echo     Cela peut prendre quelques minutes.
    echo.
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 (
        echo.
        echo [OK] Python 3.12 installe !
        echo.
        REM Rafraichir le PATH pour cette session
        for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
        for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
        set "PATH=!USER_PATH!;!SYS_PATH!"

        REM Tester python apres rafraichissement
        python --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=python"
            goto :LAUNCH_INSTALLER
        )
        py -3 --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=py -3"
            goto :LAUNCH_INSTALLER
        )

        REM Chemin direct apres install winget
        if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
            set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
            goto :LAUNCH_INSTALLER
        )

        echo [!] Python installe mais introuvable dans cette session.
        echo     Fermez cette fenetre et relancez install_windows.bat
        pause
        exit /b 1
    )
)

REM --- Methode 5 : Telecharger Python via PowerShell ---
echo [*] Tentative de telechargement de Python 3.12...
echo.
set "PY_INSTALLER=%TEMP%\python-3.12-installer.exe"
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile '%PY_INSTALLER%' }" 2>nul
if not exist "%PY_INSTALLER%" (
    echo [ERREUR] Impossible de telecharger Python automatiquement.
    echo.
    echo          Installez Python 3.12 manuellement :
    echo          https://www.python.org/downloads/
    echo.
    echo          IMPORTANT : Cochez "Add Python to PATH" pendant l'installation !
    echo          Puis relancez install_windows.bat
    pause
    exit /b 1
)

echo [*] Lancement de l'installeur Python 3.12...
echo     IMPORTANT : Cochez "Add python.exe to PATH" en bas de la fenetre !
echo.
start /wait "" "%PY_INSTALLER%" InstallAllUsers=0 PrependPath=1 Include_launcher=1
del "%PY_INSTALLER%" >nul 2>&1

REM Rafraichir le PATH
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
set "PATH=!USER_PATH!;!SYS_PATH!"

REM Verifier apres installation manuelle
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :LAUNCH_INSTALLER
)
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :LAUNCH_INSTALLER
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :LAUNCH_INSTALLER
)

echo.
echo [ERREUR] Python n'a pas pu etre detecte apres installation.
echo          Fermez cette fenetre et relancez install_windows.bat
pause
exit /b 1

:LAUNCH_INSTALLER
echo.
echo [*] Lancement de l'installeur graphique...
echo.
%PYTHON_CMD% "%~dp0installer.py"
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installeur a rencontre un probleme.
    pause
)
exit /b 0
