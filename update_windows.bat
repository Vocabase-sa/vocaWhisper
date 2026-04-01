@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================================
echo   VocaWhisper - Mise à jour
echo ============================================================
echo.

REM Vérifier si git est disponible
git --version >nul 2>&1
if errorlevel 1 (
    echo [!] Git n'est pas installe.
    echo     Installation de la mise a jour sans git...
    echo.
    goto :UPDATE_NO_GIT
)

REM Méthode 1 : Git pull
echo [*] Recuperation des mises a jour depuis GitHub...
git fetch origin
if errorlevel 1 (
    echo [ERREUR] Impossible de contacter GitHub.
    echo          Verifiez votre connexion internet.
    pause
    exit /b 1
)

REM Sauvegarder la branche actuelle
for /f "tokens=*" %%a in ('git rev-parse --abbrev-ref HEAD') do set BRANCH=%%a
echo [OK] Branche : %BRANCH%

REM Vérifier s'il y a des mises à jour
git diff --quiet HEAD origin/%BRANCH% 2>nul
if not errorlevel 1 (
    echo.
    echo [OK] VocaWhisper est deja a jour !
    pause
    exit /b 0
)

echo [*] Mise a jour en cours...
git pull origin %BRANCH%
if errorlevel 1 (
    echo [ERREUR] La mise a jour a echoue.
    echo          Verifiez qu'il n'y a pas de conflits.
    pause
    exit /b 1
)

goto :UPDATE_DEPS

:UPDATE_NO_GIT
REM Méthode 2 : Téléchargement via PowerShell (sans git)
echo [*] Telechargement de la derniere version...
powershell -Command "& { try { Invoke-WebRequest -Uri 'https://github.com/Vocabase-sa/vocaWhisper/archive/refs/heads/main.zip' -OutFile 'update.zip' -UseBasicParsing } catch { Write-Host '[ERREUR] Telechargement echoue'; exit 1 } }"
if errorlevel 1 (
    echo [ERREUR] Impossible de telecharger la mise a jour.
    pause
    exit /b 1
)

echo [*] Extraction des fichiers...
powershell -Command "Expand-Archive -Path 'update.zip' -DestinationPath 'update_tmp' -Force"

REM Copier les fichiers mis à jour (sans écraser config.json, venv, noms_propres.txt)
echo [*] Application de la mise a jour...
for %%f in (update_tmp\vocaWhisper-main\*.py update_tmp\vocaWhisper-main\*.txt update_tmp\vocaWhisper-main\*.bat update_tmp\vocaWhisper-main\*.vbs update_tmp\vocaWhisper-main\*.md) do (
    set "fname=%%~nxf"
    if /i not "!fname!"=="config.json" (
        if /i not "!fname!"=="noms_propres.txt" (
            if /i not "!fname!"=="vocabulaire.txt" (
                copy /y "%%f" "." >nul
            )
        )
    )
)

REM Copier les sous-dossiers (api, icons, etc.)
if exist "update_tmp\vocaWhisper-main\api" (
    xcopy /s /y /q "update_tmp\vocaWhisper-main\api\*" "api\" >nul 2>&1
)
if exist "update_tmp\vocaWhisper-main\icons" (
    xcopy /s /y /q "update_tmp\vocaWhisper-main\icons\*" "icons\" >nul 2>&1
)

REM Nettoyage
del /q update.zip 2>nul
rmdir /s /q update_tmp 2>nul

:UPDATE_DEPS
echo.
echo [*] Mise a jour des dependances...

REM Activer le venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [!] Environnement virtuel non trouve.
    echo     Relancez install_windows.bat pour reinstaller.
    pause
    exit /b 1
)

REM Vérifier quel mode est installé (présence de faster-whisper)
pip show faster-whisper >nul 2>&1
if errorlevel 1 (
    echo [*] Mode Groq detecte - installation des dependances de base...
    pip install -r requirements-base.txt --quiet
) else (
    echo [*] Mode complet detecte - installation de toutes les dependances...
    pip install -r requirements-base.txt --quiet
    pip install -r requirements-local.txt --quiet
)

echo.
echo ============================================================
echo   Mise a jour terminee !
echo ============================================================
echo.

REM Afficher la version
if exist "VERSION" (
    set /p NEW_VER=<VERSION
    echo   Version : !NEW_VER!
)
echo.
echo   Relancez VocaWhisper pour appliquer les changements.
echo.
pause
