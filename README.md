# VocaWhisper

Dictez du texte par la voix avec un raccourci clavier global configurable, le texte transcrit est automatiquement copie dans le presse-papier et colle dans l'application active.

Supporte deux moteurs de transcription : [faster-whisper](https://github.com/SYSTRAN/faster-whisper) en local (GPU CUDA / Apple MPS / CPU) ou [Groq](https://groq.com/) dans le cloud pour une transcription ultra-rapide.

---

## Fonctionnalites

- **Raccourcis configurables** : 2 raccourcis clavier au choix (Ctrl+Space, Ctrl+², Ctrl+F1-F5, Ctrl+Shift+D/A/Space) configurables depuis l'UI
- **Double moteur STT** : transcription locale (faster-whisper) ou cloud (Groq), avec fallback automatique optionnel
- **Collage automatique** : le texte transcrit est colle directement dans l'application active
- **Prompt initial** : ajoutez des mots techniques pour orienter la reconnaissance vocale
- **Corrections automatiques** : regles de post-traitement pour corriger les erreurs recurrentes
- **Correction fuzzy des noms propres** : corrige automatiquement les noms propres mal transcrits via [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), avec seuil configurable
- **Fine-tuning Whisper** : entrainez le modele sur vos propres donnees audio pour ameliorer la reconnaissance de votre vocabulaire
- **API HTTP** : endpoint `/transcribe` pour integrer la dictee dans d'autres applications (Flask)
- **Icone system tray** : acces rapide aux parametres et controle de l'application
- **Overlay futuriste** : pilule flottante avec indicateur REC, timer et barres audio animees en temps reel
- **Choix du microphone** : selection du peripherique audio dans les parametres
- **Demarrage automatique** : option pour lancer l'application au demarrage du systeme
- **Multi-modeles** : tiny, base, small, medium, large-v2, large-v3, large-v3-turbo, distil-large-v3
- **Fenetre de telechargement** : progression visuelle lors du telechargement initial du modele
- **Detection du cache** : affiche si chaque modele est deja telecharge ou a telecharger, avec sa taille

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
| `hotkey_primary` | Raccourci principal | `Ctrl+Space`, `Ctrl+²`, `Ctrl+F1`-`F5`, etc. |
| `hotkey_secondary` | Raccourci secondaire | Memes valeurs ou `Aucun` |
| `stt_engine` | Moteur de transcription | `local` (faster-whisper) ou `groq` (cloud) |
| `groq_api_key` | Cle API Groq | Cle obtenue sur [console.groq.com](https://console.groq.com) |
| `groq_model` | Modele Groq | `whisper-large-v3`, `whisper-large-v3-turbo` |
| `groq_fallback_local` | Fallback sur le modele local | `true` / `false` — si Groq echoue, utilise le modele local |
| `fuzzy_enabled` | Correction fuzzy des noms propres | `true` / `false` |
| `fuzzy_threshold` | Seuil de similarite fuzzy | `0`-`100` (defaut: `60`) |
| `api_enabled` | Activer l'API HTTP | `true` / `false` |
| `api_host` | Adresse d'ecoute API | `0.0.0.0` (toutes interfaces) ou `127.0.0.1` (local) |
| `api_port` | Port de l'API HTTP | defaut: `5892` |

### Modeles recommandes

| Configuration | Modele | Precision | RAM GPU | Langues |
|---------------|--------|-----------|---------|---------|
| NVIDIA RTX | `large-v3` | `float16` | ~3 Go | Multilingue |
| NVIDIA RTX (rapide) | `large-v3-turbo` | `float16` | ~1.6 Go | Multilingue |
| Apple Silicon | `large-v3-turbo` | `float16` | ~1.6 Go | Multilingue |
| CPU performant | `medium` | `int8` | - | Multilingue |
| CPU modeste | `small` | `int8` | - | Multilingue |
| Anglais uniquement | `distil-large-v3` | `float16` | ~1.5 Go | Anglais uniquement |

> **Attention** : Les modeles `distil-large-v2` et `distil-large-v3` ne supportent que **l'anglais**. Pour le francais ou toute autre langue, utilisez `large-v3` ou `large-v3-turbo`. L'application affiche un avertissement si un modele distil est selectionne avec une langue non anglaise.

---

## Fonctionnement

### Cycle de dictee

1. **Lancement** : l'application charge le moteur STT (local ou Groq) et affiche une icone verte dans le system tray
2. **Raccourci (1er appui)** : l'enregistrement audio demarre, un overlay futuriste s'affiche (pilule avec indicateur REC, timer et barres audio en temps reel)
3. **Raccourci (2e appui)** : l'enregistrement s'arrete, l'audio est amplifie selon le gain configure
4. **Transcription** : le moteur STT (local ou Groq) transcrit l'audio en texte, en utilisant le prompt initial comme contexte
5. **Corrections** : les regles de `corrections.txt` sont appliquees au texte transcrit
6. **Fuzzy noms propres** : les noms propres mal transcrits sont corriges par similarite via `noms_propres.txt`
7. **Resultat** : le texte final est copie dans le presse-papier et automatiquement colle (Ctrl+V) dans l'application active

### Raccourcis

Les raccourcis sont configurables depuis l'onglet General des parametres. Deux raccourcis disponibles (principal + secondaire).

| Raccourci (defaut) | Action |
|---------------------|--------|
| `Ctrl+Space` | Demarrer/arreter l'enregistrement (raccourci principal) |
| `Ctrl+F2` | Demarrer/arreter l'enregistrement (raccourci secondaire) |

Raccourcis disponibles : `Ctrl+Space`, `Ctrl+²`, `Ctrl+F1` a `Ctrl+F5`, `Ctrl+Shift+D`, `Ctrl+Shift+A`, `Ctrl+Shift+Space`, ou `Aucun`.

> Pour quitter l'application, utilisez le clic droit sur l'icone system tray > **Quitter**.

### Icone system tray

Un clic droit sur l'icone dans la barre des taches donne acces a :

- **Parametres** : ouvre la fenetre de configuration (onglets General, Prompt initial, Corrections, Noms propres, Training)
- **Quitter** : ferme l'application

L'icone change de couleur : **verte** = pret, **rouge** = enregistrement en cours.

#### Icone personnalisee

Vous pouvez remplacer l'icone par defaut en placant vos images dans le dossier `icons/` :

| Fichier | Usage |
|---------|-------|
| `icons/icon_green.png` | Icone quand l'app est prete |
| `icons/icon_red.png` | Icone pendant l'enregistrement |
| `icons/icon.png` | Icone unique (utilisee pour les deux etats) |

Les formats `.png` et `.ico` sont supportes. Taille recommandee : 64x64 pixels. Si aucune icone n'est trouvee, l'icone par defaut (cercle colore avec micro) est utilisee.

### Overlay d'enregistrement

Pendant l'enregistrement, une pilule futuriste flottante s'affiche en bas de l'ecran :

- **Point REC** clignotant avec halo neon
- **Timer** en temps reel (minutes:secondes)
- **Barres audio** animees qui reagissent au niveau du microphone
- **Bordure neon** pulsante (rouge sombre a rouge vif)
- L'overlay est click-through (les clics passent a travers)

---

## Parametres (interface graphique)

Accessible via clic droit sur l'icone tray > **Parametres**. La fenetre comporte 5 onglets :

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
| **Raccourci 1 / 2** | Raccourcis clavier configurables | Choisissez parmi la liste (Ctrl+Space, Ctrl+², Ctrl+F1-F5, etc.). Un changement necessite un redemarrage. |
| **Moteur STT** | Moteur de transcription | `local` (faster-whisper) ou `groq` (cloud). Voir section Groq ci-dessous. |

L'onglet General affiche aussi :
- Un indicateur en temps reel pour chaque modele : **en local** (avec la taille) ou **a telecharger** (avec la taille estimee)
- Un **avertissement** si un modele anglais uniquement est selectionne avec une autre langue

#### Configuration Groq (cloud)

Quand le moteur STT est regle sur `groq` :
- **Cle API Groq** : votre cle API (obtenue sur [console.groq.com](https://console.groq.com))
- **Modele Groq** : `whisper-large-v3` (meilleure qualite) ou `whisper-large-v3-turbo` (plus rapide)
- **Fallback local** : si active, utilise le modele local en cas d'echec de Groq (erreur reseau, quota, etc.)

> **Avantage Groq** : demarrage quasi-instantane (pas de chargement de modele GPU), transcription tres rapide. Necessite une connexion internet.

> **Fallback** : si le fallback est desactive, le modele local n'est pas charge en memoire, ce qui reduit drastiquement le temps de demarrage et la consommation VRAM.

### Onglet Prompt initial

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
camaieu -> Kamailio

# Corrections infra
cubernétise -> Kubernetes
gite -> Git
```

> **Astuce** : dictez vos termes techniques plusieurs fois, notez les erreurs recurrentes de Whisper, puis ajoutez les corrections correspondantes. Combinez avec le prompt initial pour un maximum de precision.

### Onglet Noms propres

Gere la correction automatique des noms propres par similarite (fuzzy matching via [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)).

| Parametre | Description |
|-----------|-------------|
| **Activer la correction fuzzy** | Active/desactive la correction automatique des noms propres |
| **Seuil de similarite** | Score minimum (0-100) pour accepter une correction. Defaut : 60. Plus bas = plus tolerant, plus de risques de faux positifs. |
| **Liste de noms** | Un nom propre par ligne (noms simples ou composes). Les lignes `#` sont ignorees. |

**Comment ca marche** : apres la transcription et les corrections regex, chaque mot du texte est compare aux noms de la liste. Si la similarite depasse le seuil, le mot est remplace. Les noms composes (ex: "Abugattas de Torres") sont traites en priorite.

Exemple (`noms_propres.txt`) :
```
# Medecins
Jamoulle
Rochdi
Abugattas de Torres

# Lieux
rondenbosh
```

> **Astuce** : si Groq ou Whisper transcrit systematiquement un nom propre de facon incorrecte mais proche (ex: "Rondenboche" au lieu de "rondenbosh"), ajoutez le nom correct ici. Le fuzzy matching corrigera automatiquement les variantes.

### Onglet Training (Fine-tuning)

Permet d'entrainer le modele Whisper sur vos propres enregistrements audio pour ameliorer la reconnaissance de votre vocabulaire specifique (termes techniques, noms propres, jargon metier).

> **Note** : Au premier lancement depuis l'onglet Training, les dependances supplementaires (~5 Go) sont installees automatiquement. Vous n'avez rien a faire, attendez simplement que l'installation se termine dans le Journal.

#### Principe

Le fine-tuning reentraine le modele Whisper avec vos propres enregistrements vocaux. Plus vous fournissez d'exemples audio de votre vocabulaire specifique, mieux le modele le reconnaitra. Le processus se fait en 3 etapes depuis l'interface, sans aucune ligne de commande.

#### Etape 1 : Preparer les donnees

1. **Enregistrez** des phrases contenant votre vocabulaire specifique (fichiers `.wav` ou `.mp3`)
2. **Placez** les fichiers audio dans le dossier `fine_tuning/data/audio/`
3. **Creez** un fichier CSV `fine_tuning/data/transcriptions.csv` avec le format :

```csv
audio_file,transcription
001.wav,"Bonjour, je teste le fine-tuning de Whisper."
002.wav,"OpenSIPs est un serveur SIP open source."
003.wav,"Frederic Jamoulle travaille sur le projet Vocabase."
```

4. Dans l'onglet Training, verifiez que les champs **Fichier CSV** et **Dossier audio** pointent vers vos fichiers
5. Cliquez sur **Preparer le dataset**

Le Journal affiche la progression. Une fois termine, vos donnees sont pretes pour l'entrainement.

> **Combien de donnees ?** Minimum **50 phrases** pour des resultats visibles. Avec 5-10 phrases, le modele fonctionne mais les ameliorations seront limitees. Idealement, visez **100+ phrases** couvrant votre vocabulaire.

#### Etape 2 : Lancer l'entrainement

| Parametre | Description | Valeur recommandee |
|-----------|-------------|--------------------|
| **Modele de base** | Modele Whisper a fine-tuner | `bofenghuang/whisper-large-v3-french` (francais) ou `openai/whisper-large-v3` (multilingue) |
| **Epoques** | Nombre de passes sur les donnees | `3` (peu de donnees) a `10` (beaucoup de donnees) |
| **Batch size** | Exemples traites par iteration | `4` (16 Go VRAM) ou `8` (24 Go VRAM) |
| **Learning rate** | Vitesse d'apprentissage | `1e-5` (recommande, ne pas modifier sauf si necessaire) |

Cliquez sur **Lancer l'entrainement**. Le Journal affiche :
- La configuration utilisee (modele, device, VRAM, etc.)
- Le chargement du dataset et du processeur audio
- La preparation des features audio (barre de progression `Map`)
- L'avancement de chaque epoque avec le **loss** (erreur du modele, doit diminuer)
- La sauvegarde du modele final

> **Duree** : Environ 30 secondes par epoque pour 5 exemples sur RTX 4090. Le temps augmente lineairement avec le nombre d'exemples.

L'encoder est automatiquement gele pour eviter le "catastrophic forgetting" (perte des connaissances du modele original). Seul le decoder est fine-tune, ce qui est suffisant pour apprendre votre vocabulaire.

#### Etape 3 : Convertir le modele

Une fois l'entrainement termine, le modele doit etre converti au format **CTranslate2** pour etre utilise par Vocabase (faster-whisper).

1. Selectionnez la quantization **float16** (recommande pour GPU, reduit la taille du modele de moitie sans perte de qualite)
2. Cliquez sur **Convertir**
3. Le Journal affiche la progression de la conversion

| Quantization | Usage | Taille approximative |
|-------------|-------|---------------------|
| **float16** | GPU (NVIDIA, Apple Silicon) — **recommande** | ~3 Go |
| **float32** | Maximum de precision (rarement necessaire) | ~6 Go |
| **int8** | CPU (plus compact, leger) | ~1.5 Go |

#### Etape 4 : Utiliser le modele fine-tune

1. Le modele converti se trouve dans `fine_tuning/model_ct2/`
2. Ouvrez l'onglet **General** des parametres
3. Dans le champ **Modele personnalise**, entrez le chemin : `fine_tuning/model_ct2`
4. Cliquez sur **Sauvegarder** puis **Redemarrer**

Vocabase utilisera desormais votre modele fine-tune. Pour revenir au modele standard, videz le champ **Modele personnalise** et redemarrez.

#### Utilisation en ligne de commande (optionnel)

Le fine-tuning peut aussi etre lance en ligne de commande :

```bash
# Activer l'environnement virtuel
# Windows : venv\Scripts\activate.bat
# macOS/Linux : source venv/bin/activate

# 1. Preparer le dataset
python fine_tuning/prepare_dataset.py

# 2. Entrainer (GPU NVIDIA recommande)
python fine_tuning/train.py --epochs 3 --batch_size 8

# 3. Convertir en CTranslate2
python fine_tuning/convert_to_ct2.py --quantization float16
```

#### Compatibilite

| Plateforme | Device | Support |
|------------|--------|---------|
| Windows + NVIDIA | CUDA | Pleinement supporte |
| Linux + NVIDIA | CUDA | Pleinement supporte |
| macOS Apple Silicon | MPS | Supporte (plus lent) |
| Tout OS sans GPU | CPU | Supporte (lent, petits datasets uniquement) |

---

## Structure du projet

```
vocaWhisper/
├── whisper_dictation.py   # Application principale
├── config_ui.py           # Interface de parametres (tkinter)
├── fuzzy_correction.py    # Correction fuzzy des noms propres (rapidfuzz)
├── overlay_ui.py          # Overlay futuriste d'enregistrement
├── download_ui.py         # Fenetre de progression du telechargement
├── api/                   # API HTTP (Flask)
│   ├── server.py          # Serveur Flask (/transcribe, /health)
│   └── __init__.py
├── config.json.example    # Exemple de configuration
├── vocabulaire.txt        # Prompt initial pour Whisper
├── corrections.txt        # Regles de correction post-transcription
├── noms_propres.txt       # Noms propres pour correction fuzzy
├── requirements.txt       # Dependances Python
├── fine_tuning/           # Pipeline de fine-tuning Whisper
│   ├── prepare_dataset.py # Preparation du dataset audio
│   ├── train.py           # Entrainement du modele
│   ├── convert_to_ct2.py  # Conversion vers CTranslate2
│   ├── requirements.txt   # Dependances fine-tuning
│   └── data/              # Donnees d'entrainement (audio + CSV)
├── icons/                 # Icones personnalisees (icon_green.png, icon_red.png)
├── install_windows.bat    # Script d'installation Windows
├── install_mac.sh         # Script d'installation macOS
├── run.bat                # Lanceur Windows (avec console)
├── run_silent.vbs         # Lanceur Windows (sans fenetre)
└── .gitignore
```

---

## API HTTP

VocaWhisper expose une API REST (Flask) pour integrer la transcription dans d'autres applications.

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/transcribe` | POST | Envoie un fichier audio et recoit la transcription |
| `/health` | GET | Etat du serveur (moteur STT, pret ou non) |

Activable dans les parametres (onglet General > API). Par defaut sur le port `5892`.

Exemple :
```bash
curl -X POST http://localhost:5892/transcribe \
  -F "audio=@mon_fichier.wav"
```

---

## Logs

Tous les logs sont centralises dans le dossier `logs/` :

- `logs/whisper_dictation.log` — logs de l'application de dictee (erreurs micro, GPU, modele, etc.)
- `logs/transcribe.log` — progression du batch de transcription
- `logs/transcribe_err.log` — erreurs du batch de transcription

---

## Depannage

| Probleme | Solution |
|----------|----------|
| `CUDA not available` | Verifiez que les drivers NVIDIA et CUDA toolkit sont installes |
| Micro non detecte | Verifiez les permissions micro dans les parametres systeme |
| Transcription lente | Passez a un modele plus petit (`small`, `medium`), verifiez que le GPU est actif, ou passez sur Groq (cloud) |
| Texte non colle | Verifiez que `auto_paste` est active et que l'application cible accepte Ctrl+V |
| Parametres ne s'ouvrent pas | Verifiez les logs ; l'interface se lance dans un processus separe |
| Modele distil ne transcrit pas le francais | Les modeles `distil` ne supportent que l'anglais. Utilisez `large-v3` ou `large-v3-turbo` |
| Modele affiche "a telecharger" alors qu'il est present | Le nom du cache peut varier selon l'org. Mettez a jour vers la derniere version |
| L'app se ferme toute seule | Verifiez les logs (`logs/whisper_dictation.log`) pour identifier la cause |
| Raccourci ne fonctionne pas | Verifiez le raccourci configure dans les parametres. Les touches F1-F5 sont les plus fiables. Ctrl+² necessite un clavier AZERTY. |
| Groq echoue | Verifiez votre cle API et votre connexion internet. Activez le fallback local si necessaire. |
| Noms propres mal corriges | Ajustez le seuil fuzzy (plus bas = plus tolerant). Verifiez que le nom est bien dans `noms_propres.txt`. |

---

## Licence

Ce projet est un outil interne. Contactez les mainteneurs pour les conditions d'utilisation.
