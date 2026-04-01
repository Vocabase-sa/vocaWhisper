#!/bin/bash
# ============================================================
#   VocaWhisper - Installation macOS
# ============================================================

set -e

echo "============================================================"
echo "  VocaWhisper - Vérification de Python"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Vérifier Python 3 ---
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "[OK] $PYTHON_VERSION détecté"
else
    echo "[!] Python 3 n'est pas installé."
    echo ""

    # Tenter Homebrew
    if command -v brew &> /dev/null; then
        echo "[*] Installation de Python via Homebrew..."
        brew install python@3.12
    else
        echo "[!] Homebrew non détecté. Installation de Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo "[*] Installation de Python via Homebrew..."
        brew install python@3.12
    fi

    # Vérifier à nouveau
    if ! command -v python3 &> /dev/null; then
        echo "[ERREUR] Python 3 n'a pas pu être installé."
        echo "         Installez-le manuellement : brew install python3"
        echo "         Ou téléchargez depuis https://python.org"
        exit 1
    fi

    echo "[OK] $(python3 --version) installé avec succès."
fi

echo ""
echo "[*] Lancement de l'installeur graphique..."
echo ""
python3 "$SCRIPT_DIR/installer.py"
