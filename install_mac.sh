#!/bin/bash
# ============================================================
#   Whisper Dictation - Installation macOS
# ============================================================

set -e

echo "============================================================"
echo "  Whisper Dictation - Installation macOS"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Vérifier Python 3 ---
if ! command -v python3 &> /dev/null; then
    echo "[ERREUR] Python 3 n'est pas installé."
    echo "Installe-le avec : brew install python3"
    echo "Ou télécharge depuis https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] $PYTHON_VERSION détecté"
echo ""

# --- Créer le venv ---
if [ ! -d "venv" ]; then
    echo "[*] Création de l'environnement virtuel..."
    python3 -m venv venv
    echo "[OK] Environnement virtuel créé."
else
    echo "[OK] Environnement virtuel existant détecté."
fi

# Activer le venv
source venv/bin/activate
echo ""

# --- Détecter l'architecture (Apple Silicon vs Intel) ---
ARCH=$(uname -m)
echo "[*] Architecture détectée : $ARCH"
echo ""

if [ "$ARCH" = "arm64" ]; then
    echo "============================================================"
    echo "  Mac Apple Silicon (M1/M2/M3/M4) détecté"
    echo "  PyTorch utilisera l'accélération MPS (Metal)"
    echo "============================================================"
    echo ""
    echo "[*] Installation de PyTorch (avec support MPS)..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    DEVICE="mps"
    COMPUTE="float16"
    RECOMMENDED_MODEL="large-v3-turbo"
else
    echo "============================================================"
    echo "  Mac Intel détecté"
    echo "  PyTorch utilisera le CPU uniquement"
    echo "============================================================"
    echo ""
    echo "[*] Installation de PyTorch (CPU)..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    DEVICE="cpu"
    COMPUTE="int8"
    RECOMMENDED_MODEL="small"
fi

echo ""
echo "[*] Installation des dépendances..."
pip install faster-whisper sounddevice numpy pyperclip pynput pystray Pillow

echo ""

# --- Créer le config.json si absent ---
if [ ! -f "config.json" ]; then
    echo "[*] Création de la configuration par défaut..."
    cat > config.json << EOF
{
    "model_size": "$RECOMMENDED_MODEL",
    "device": "$DEVICE",
    "compute_type": "$COMPUTE",
    "language": "fr",
    "audio_gain": 10.0,
    "auto_paste": true,
    "auto_start": false,
    "microphone": ""
}
EOF
    echo "[OK] config.json créé (modèle: $RECOMMENDED_MODEL, device: $DEVICE)"
else
    echo "[OK] config.json existant conservé."
fi

echo ""

# --- Créer le script de lancement ---
cat > run_mac.command << 'RUNEOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate
python3 whisper_dictation.py
RUNEOF
chmod +x run_mac.command

echo ""
echo "============================================================"
echo "  Installation terminée !"
echo "============================================================"
echo ""
if [ "$ARCH" = "arm64" ]; then
    echo "  Mode : Apple Silicon (MPS - accélération Metal)"
    echo "  Modèle recommandé : large-v3-turbo ou large-v3"
else
    echo "  Mode : Intel Mac (CPU)"
    echo "  Modèle recommandé : small ou medium"
fi
echo ""
echo "  Pour lancer : double-clique sur run_mac.command"
echo "  Ou en terminal : ./run_mac.command"
echo ""
echo "  NOTE: macOS va peut-être demander l'autorisation"
echo "  d'accès au micro et aux touches (Accessibilité)."
echo "  Accepte ces permissions dans Préférences Système."
echo ""
