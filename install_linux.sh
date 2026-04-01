#!/bin/bash
# ============================================================
#   VocaWhisper - Installation Linux
# ============================================================

set -e

echo "============================================================"
echo "  VocaWhisper - Vérification des dépendances"
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

    # Détecter le gestionnaire de paquets
    if command -v apt &> /dev/null; then
        echo "[*] Installation via apt (Debian/Ubuntu)..."
        sudo apt update
        sudo apt install -y python3 python3-venv python3-pip python3-tk
    elif command -v dnf &> /dev/null; then
        echo "[*] Installation via dnf (Fedora)..."
        sudo dnf install -y python3 python3-tkinter
    elif command -v pacman &> /dev/null; then
        echo "[*] Installation via pacman (Arch)..."
        sudo pacman -S --noconfirm python python-pip tk
    else
        echo "[ERREUR] Gestionnaire de paquets non reconnu."
        echo "         Installez Python 3.10+ manuellement."
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        echo "[ERREUR] Python 3 n'a pas pu être installé."
        exit 1
    fi

    echo "[OK] $(python3 --version) installé avec succès."
fi

# --- Vérifier tkinter ---
echo ""
echo "[*] Vérification de tkinter..."
if python3 -c "import tkinter" 2>/dev/null; then
    echo "[OK] tkinter disponible"
else
    echo "[!] tkinter manquant. Installation..."
    if command -v apt &> /dev/null; then
        sudo apt install -y python3-tk
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-tkinter
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm tk
    fi
fi

# --- Vérifier les dépendances audio ---
echo ""
echo "[*] Vérification des dépendances audio..."
if ! command -v portaudio &> /dev/null && ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
    echo "[!] PortAudio manquant. Installation..."
    if command -v apt &> /dev/null; then
        sudo apt install -y portaudio19-dev
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y portaudio-devel
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm portaudio
    fi
fi

# --- Vérifier xclip/wl-clipboard ---
if [ -n "$WAYLAND_DISPLAY" ]; then
    if ! command -v wl-copy &> /dev/null; then
        echo "[!] wl-clipboard manquant (requis pour le presse-papier Wayland)."
        if command -v apt &> /dev/null; then
            sudo apt install -y wl-clipboard
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y wl-clipboard
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm wl-clipboard
        fi
    fi
else
    if ! command -v xclip &> /dev/null; then
        echo "[!] xclip manquant (requis pour le presse-papier X11)."
        if command -v apt &> /dev/null; then
            sudo apt install -y xclip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y xclip
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm xclip
        fi
    fi
fi

echo ""
echo "[*] Lancement de l'installeur graphique..."
echo ""
python3 "$SCRIPT_DIR/installer.py"
