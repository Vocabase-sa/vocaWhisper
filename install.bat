@echo off
echo ============================================================
echo   Installation de Whisper Dictation
echo ============================================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou pas dans le PATH.
    echo Télécharge Python 3.10+ depuis https://python.org
    pause
    exit /b 1
)

REM Créer un environnement virtuel
if not exist "venv" (
    echo [*] Création de l'environnement virtuel...
    python -m venv venv
)

REM Activer le venv
call venv\Scripts\activate.bat

REM Installer PyTorch avec CUDA (RTX 4090)
echo.
echo [*] Installation de PyTorch avec support CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Installer les dépendances
echo.
echo [*] Installation des dépendances...
pip install -r requirements.txt

echo.
echo ============================================================
echo   Installation terminée !
echo ============================================================
echo.
echo Pour lancer le programme :
echo   1. venv\Scripts\activate.bat
echo   2. python whisper_dictation.py
echo.
echo Ou utilise simplement : run.bat
echo.
pause
