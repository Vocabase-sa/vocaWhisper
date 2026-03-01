# VocaWhisper

Clone open-source de [SuperWhisper](https://superwhisper.com/) pour **Windows**, **macOS** et **Linux**.

Dictez du texte par la voix avec un raccourci clavier global (**Ctrl+Space**), le texte transcrit est automatiquement copie dans le presse-papier et colle dans l'application active.

Utilise [faster-whisper](https://github.com/SYSTRAN/faster-whisper) avec acceleration GPU (CUDA / Apple MPS) ou CPU.

---

## Fonctionnalites

- **Raccourci global Ctrl+Space** : demarrer/arreter l'enregistrement depuis n'importe quelle application
- **Transcription locale** : tout reste sur votre machine, aucune donnee envoyee dans le cloud
- **Collage automatique** : le texte transcrit est colle directement dans l'application active
- **Vocabulaire personnalise** : ajoutez des mots techniques pour ameliorer la reconnaissance
- **Corrections automatiques** : regles de post-traitement pour corriger les erreurs recurrentes
- **Icone system tray** : acces rapide aux parametres et controle de l'application
- **Overlay visuel** : pastille rouge clignotante pendant l'enregistrement
- **Choix du microphone** : selection du peripherique audio dans les parametres
- **Demarrage automatique** : option pour lancer l'application au demarrage du systeme
- **Multi-modeles** : tiny, base, small, medium, large-v2, large-v3, large-v3-turbo, distil-large-v3
- **Fenetre de telechargement** : progression visuelle lors du telechargement initial du modele

---

## Pre-requis

| Element | Version |
|---------|---------|
| Python | 3.10 ou superieur |
| OS | Windows 10/11, macOS 12+, Linux (X11/Wayland) |
| GPU (optionnel) | NVIDIA GTX/RTX (CUDA) ou Apple Silicon M1/M2/M3/M4 (MPS) |

> **Note** : Un GPU est fortement recommande pour les modeles `large`. Sur CPU, privilegiez les modeles `small` ou `medium`.

---

## Installation

### Windows

#### Methode rapide (recommandee)

```
git clone https://github.com/Vocabase-sa/vocaWhisper.git
cd vocaWhisper
install_windows.bat
```

Double-cliquez sur `install_windows.bat` et suivez les instructions :
1. Le script verifie que Python est installe
2. Cree un environnement virtuel (`venv`)
3. Demande si vous avez un GPU NVIDIA (choix CUDA ou CPU)
4. Installe PyTorch et toutes les dependances

#### Methode manuelle

```bash
git clone https://github.com/Vocabase-sa/vocaWhisper.git
cd vocaWhisper
python -m venv venv
venv\Scripts\activate.bat

# Avec GPU NVIDIA :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Ou sans GPU :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install pynput
```

#### Lancement (Windows)

| Methode | Description |
|---------|-------------|
| `run.bat` | Lance avec fenetre console (utile pour le debug) |
| `run_silent.vbs` | Lance sans fenetre (recommande en usage quotidien) |
| Demarrage auto | Activable dans les parametres (raccourci Startup) |

---

### macOS

```bash
git clone https://github.com/Vocabase-sa/vocaWhisper.git
cd vocaWhisper
chmod +x install_mac.sh
./install_mac.sh
```

Le script detecte automatiquement l'architecture :
- **Apple Silicon (M1/M2/M3/M4)** : installe PyTorch avec MPS (Metal), modele recommande `large-v3-turbo`
- **Intel Mac** : installe PyTorch CPU, modele recommande `small`

#### Lancement (macOS)

```bash
./run_mac.command
# Ou double-cliquez sur run_mac.command dans le Finder
```

> **Permissions requises** : macOS demandera l'autorisation d'acces au **microphone** et aux **touches** (Accessibilite). Acceptez ces permissions dans Preferences Systeme > Securite et confidentialite.

---

### Linux

```bash
git clone https://github.com/Vocabase-sa/vocaWhisper.git
cd vocaWhisper
python3 -m venv venv
source venv/bin/activate

# Avec GPU NVIDIA :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Ou sans GPU :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install pynput
```

#### Dependances systeme (Linux)

Selon votre distribution, vous pourriez avoir besoin de :

```bash
# Debian / Ubuntu
sudo apt install python3-tk portaudio19-dev xclip

# Fedora
sudo dnf install python3-tkinter portaudio-devel xclip

# Arch Linux
sudo pacman -S tk portaudio xclip
```

- `python3-tk` : interface de parametres (tkinter)
- `portaudio` : capture audio (requis par sounddevice)
- `xclip` : presse-papier (requis par pyperclip sur X11)

#### Lancement (Linux)

```bash
source venv/bin/activate
python3 whisper_dictation.py
```

> **Note** : Sur Wayland, `xclip` peut ne pas fonctionner. Installez `wl-clipboard` a la place et configurez pyperclip en consequence.

---

## Configuration

### Fichier config.json

Au premier lancement, copiez le fichier d'exemple :

```bash
cp config.json.example config.json
```

Ou utilisez les parametres via l'icone system tray (clic droit > Parametres).

| Parametre | Description | Valeurs |
|-----------|-------------|---------|
| `model_size` | Modele Whisper | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v3` |
| `device` | Peripherique de calcul | `cuda` (NVIDIA), `mps` (Apple Silicon), `cpu` |
| `compute_type` | Precision numerique | `float16` (GPU), `float32`, `int8` (CPU) |
| `language` | Langue de transcription | `fr`, `en`, `de`, `es`, `nl`, `it`, `pt`, `null` (auto) |
| `audio_gain` | Amplification micro | `1.0` a `20.0` (defaut: `10.0`) |
| `auto_paste` | Coller automatiquement | `true` / `false` |
| `auto_start` | Demarrage avec le systeme | `true` / `false` |
| `microphone` | Nom du micro | `""` (defaut systeme) ou nom du peripherique |

### Modeles recommandes

| Configuration | Modele | Precision | RAM GPU |
|---------------|--------|-----------|---------|
| NVIDIA RTX | `large-v3` | `float16` | ~3 Go |
| NVIDIA RTX (rapide) | `large-v3-turbo` | `float16` | ~1.6 Go |
| Apple Silicon | `large-v3-turbo` | `float16` | ~1.6 Go |
| CPU performant | `medium` | `int8` | - |
| CPU modeste | `small` | `int8` | - |

---

## Fonctionnement

### Cycle de dictee

1. **Lancement** : l'application charge le modele Whisper en memoire (GPU ou CPU) et affiche une icone verte dans le system tray
2. **Ctrl+Space (1er appui)** : l'enregistrement audio demarre, un overlay futuriste s'affiche (pilule avec indicateur REC, timer et barres audio en temps reel)
3. **Ctrl+Space (2e appui)** : l'enregistrement s'arrete, l'audio est amplifie selon le gain configure
4. **Transcription** : faster-whisper transcrit l'audio en texte, en utilisant le vocabulaire personnalise comme contexte
5. **Corrections** : les regles de `corrections.txt` sont appliquees au texte transcrit
6. **Resultat** : le texte final est copie dans le presse-papier et automatiquement colle (Ctrl+V) dans l'application active

### Raccourcis

| Raccourci | Action |
|-----------|--------|
| `Ctrl+Space` | Demarrer/arreter l'enregistrement |
| `Echap` | Quitter l'application |

### Icone system tray

Un clic droit sur l'icone dans la barre des taches donne acces a :

- **Parametres** : ouvre la fenetre de configuration (onglets General, Vocabulaire, Corrections)
- **Quitter** : ferme l'application

L'icone change de couleur : **verte** = pret, **rouge** = enregistrement en cours.

---

## Parametres (interface graphique)

Accessible via clic droit sur l'icone tray > **Parametres**. La fenetre comporte 3 onglets :

### Onglet General

| Parametre | Description | Detail |
|-----------|-------------|--------|
| **Modele Whisper** | Taille du modele de reconnaissance | `tiny` (rapide, peu precis) a `large-v3` (lent, tres precis). Un changement necessite un redemarrage. |
| **Device** | Peripherique de calcul | `cuda` (GPU NVIDIA), `mps` (Apple Silicon) ou `cpu`. Un changement necessite un redemarrage. |
| **Precision** | Type de calcul numerique | `float16` (GPU, rapide), `float32` (precis), `int8` (CPU, economique). Un changement necessite un redemarrage. |
| **Langue** | Langue de transcription | `fr`, `en`, `de`, `es`, `nl`, `it`, `pt` ou `auto` (detection automatique) |
| **Gain micro** | Amplification de l'audio | Curseur de x1.0 a x20.0. Augmentez si votre micro est trop faible. |
| **Microphone** | Peripherique d'entree audio | Liste les micros detectes. "(defaut systeme)" utilise le micro par defaut de l'OS. |
| **Coller automatiquement** | Coller le texte apres transcription | Si active, simule Ctrl+V apres la copie dans le presse-papier |
| **Demarrage automatique** | Lancer avec le systeme | Windows : cree un raccourci dans le dossier Startup. macOS : cree un LaunchAgent. |

L'onglet General affiche aussi un indicateur en temps reel pour chaque modele : **en local** (avec la taille) ou **a telecharger** (avec la taille estimee).

### Onglet Vocabulaire

Zone de texte libre pour ajouter des mots ou expressions que Whisper a du mal a reconnaitre. Un mot ou expression par ligne, les lignes commencant par `#` sont ignorees.

**Comment ca marche** : les mots sont concatenes et envoyes comme `initial_prompt` a Whisper. Cela oriente le modele vers ces termes sans le forcer, ce qui ameliore la reconnaissance du jargon technique, des noms propres ou des acronymes.

Exemple :
```
# VoIP
OpenSIPs
Kamailio
FreeSWITCH

# Infra
Kubernetes
Docker
PostgreSQL
```

### Onglet Corrections

Zone de texte libre pour definir des regles de remplacement appliquees **apres** chaque transcription. Format : `erreur -> correction` (une par ligne, `#` pour commenter).

**Comment ca marche** : chaque regle est une recherche/remplacement insensible a la casse. Utile quand Whisper produit systematiquement le meme mot incorrect pour un terme technique.

Exemple :
```
# Whisper entend "au cercle" au lieu de "OCM"
au cercle -> OCM

# Corrections VoIP
open cypes -> OpenSIPs
camaïeu -> Kamailio

# Corrections infra
cubernétise -> Kubernetes
gîte -> Git
```

> **Astuce** : dictez vos termes techniques plusieurs fois, notez les erreurs recurrentes de Whisper, puis ajoutez les corrections correspondantes. Combinez avec le vocabulaire pour un maximum de precision.

---

## Structure du projet

```
vocaWhisper/
├── whisper_dictation.py   # Application principale
├── config_ui.py           # Interface de parametres (tkinter)
├── overlay_ui.py          # Overlay visuel d'enregistrement
├── download_ui.py         # Fenetre de progression du telechargement
├── config.json.example    # Exemple de configuration
├── vocabulaire.txt        # Mots personnalises pour Whisper
├── corrections.txt        # Regles de correction post-transcription
├── requirements.txt       # Dependances Python
├── install_windows.bat    # Script d'installation Windows
├── install_mac.sh         # Script d'installation macOS
├── run.bat                # Lanceur Windows (avec console)
├── run_silent.vbs         # Lanceur Windows (sans fenetre)
└── .gitignore
```

---

## Logs

Les logs sont ecrits dans `whisper_dictation.log` a la racine du projet. Ce fichier est utile pour diagnostiquer les problemes (erreurs micro, GPU, modele, etc.).

---

## Depannage

| Probleme | Solution |
|----------|----------|
| `CUDA not available` | Verifiez que les drivers NVIDIA et CUDA toolkit sont installes |
| Micro non detecte | Verifiez les permissions micro dans les parametres systeme |
| Transcription lente | Passez a un modele plus petit (`small`, `medium`) ou verifiez que le GPU est actif |
| Texte non colle | Verifiez que `auto_paste` est active et que l'application cible accepte Ctrl+V |
| Parametres ne s'ouvrent pas | Verifiez les logs ; l'interface se lance dans un processus separe |
| `RuntimeError: main thread` | Mettez a jour vers la derniere version (fix inclus) |

---

## Licence

Ce projet est un outil interne. Contactez les mainteneurs pour les conditions d'utilisation.
