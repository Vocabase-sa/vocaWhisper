@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo   Whisper Dictation - Installation Windows
echo ============================================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telecharge Python 3.10+ depuis https://python.org
    echo Coche bien "Add Python to PATH" pendant l'installation.
    pause
    exit /b 1
)

echo [OK] Python detecte :
python --version
echo.

REM Créer un environnement virtuel
if not exist "venv" (
    echo [*] Creation de l'environnement virtuel...
    python -m venv venv
    echo [OK] Environnement virtuel cree.
) else (
    echo [OK] Environnement virtuel existant detecte.
)
echo.

REM Activer le venv
call venv\Scripts\activate.bat

REM Demander si NVIDIA
echo ============================================================
echo   As-tu une carte graphique NVIDIA (GTX/RTX) ?
echo ============================================================
echo.
echo   1) Oui - NVIDIA GPU (recommande si tu as une GTX/RTX)
echo      Plus rapide, utilise CUDA pour l'acceleration GPU.
echo.
echo   2) Non - CPU uniquement
echo      Fonctionne sur tous les PC, mais la transcription
echo      sera plus lente. Utilise un modele plus petit (small/medium).
echo.
set /p GPU_CHOICE="Ton choix (1 ou 2) : "

echo.
if "%GPU_CHOICE%"=="1" (
    echo [*] Installation de PyTorch avec support CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo.
    echo [*] Configuration pour GPU...
    REM Le config.json par défaut utilise déjà cuda/float16
) else (
    echo [*] Installation de PyTorch CPU uniquement...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo.
    echo [*] Configuration pour CPU...
    REM Créer un config.json adapté au CPU si pas déjà présent
    if not exist "config.json" (
        echo {"model_size": "small", "device": "cpu", "compute_type": "int8", "language": "fr", "audio_gain": 10.0, "auto_paste": true, "auto_start": false, "microphone": ""} > config.json
    )
)

echo.
echo [*] Installation des dependances...
pip install -r requirements.txt

REM Installer pynput (meilleur que keyboard, pas besoin d'admin)
pip install pynput

echo.
echo ============================================================
echo   Installation terminee !
echo ============================================================
echo.
if "%GPU_CHOICE%"=="1" (
    echo   Mode : GPU NVIDIA (CUDA^)
    echo   Modele recommande : large-v3 ou large-v3-turbo
) else (
    echo   Mode : CPU uniquement
    echo   Modele recommande : small ou medium
    echo   (large-v3 sera tres lent sur CPU^)
)
echo.
echo   Pour lancer : double-clique sur run.bat
echo   Pour les parametres : clic droit sur l'icone dans le tray
echo.
pause
