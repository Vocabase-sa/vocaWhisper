#!/bin/bash
# ============================================================
#   VocaWhisper - Mise à jour (macOS / Linux)
# ============================================================

set -e

echo "============================================================"
echo "  VocaWhisper - Mise à jour"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Méthode 1 : Git ---
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "[*] Récupération des mises à jour depuis GitHub..."
    git fetch origin

    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "[OK] Branche : $BRANCH"

    if git diff --quiet HEAD "origin/$BRANCH" 2>/dev/null; then
        echo ""
        echo "[OK] VocaWhisper est déjà à jour !"
        exit 0
    fi

    echo "[*] Mise à jour en cours..."
    git pull origin "$BRANCH"
else
    # --- Méthode 2 : Download ZIP ---
    echo "[!] Git non disponible, téléchargement du ZIP..."
    curl -L -o update.zip "https://github.com/Vocabase-sa/vocaWhisper/archive/refs/heads/main.zip"
    unzip -o update.zip -d update_tmp

    # Copier les fichiers (sans écraser config.json, noms_propres.txt, vocabulaire.txt)
    echo "[*] Application de la mise à jour..."
    cd update_tmp/vocaWhisper-main
    for f in *.py *.txt *.bat *.sh *.vbs *.md *.command; do
        [ -f "$f" ] || continue
        case "$f" in
            config.json|noms_propres.txt|vocabulaire.txt) continue ;;
        esac
        cp "$f" "$SCRIPT_DIR/"
    done
    [ -d "api" ] && cp -r api/ "$SCRIPT_DIR/api/"
    [ -d "icons" ] && cp -r icons/ "$SCRIPT_DIR/icons/"
    cd "$SCRIPT_DIR"

    rm -rf update.zip update_tmp
fi

# --- Mise à jour des dépendances ---
echo ""
echo "[*] Mise à jour des dépendances..."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "[ERREUR] Environnement virtuel non trouvé."
    echo "         Relancez l'installateur."
    exit 1
fi

# Détecter le mode installé
if pip show faster-whisper &> /dev/null; then
    echo "[*] Mode complet détecté..."
    pip install -r requirements-base.txt --quiet
    pip install -r requirements-local.txt --quiet
else
    echo "[*] Mode Groq détecté..."
    pip install -r requirements-base.txt --quiet
fi

echo ""
echo "============================================================"
echo "  Mise à jour terminée !"
echo "============================================================"
if [ -f "VERSION" ]; then
    echo "  Version : $(cat VERSION)"
fi
echo ""
echo "  Relancez VocaWhisper pour appliquer les changements."
echo ""
