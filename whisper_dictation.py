"""
Whisper Dictation - Clone de SuperWhisper (Windows & macOS)
===========================================================
Ctrl+Space : Démarrer/Arrêter l'enregistrement
Le texte transcrit est automatiquement copié dans le presse-papier.

Utilise faster-whisper avec accélération GPU (CUDA/MPS) ou CPU.
"""

import collections
import platform
import re
import subprocess
import threading
import time
import sys
import os
import json
import logging

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# Sur Windows, définir un AppUserModelID pour notre icône dans la barre des tâches
if IS_WINDOWS:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vocabase.vocawhisper")

import numpy as np
import sounddevice as sd
import pyperclip
from faster_whisper import WhisperModel
from overlay_ui import show_overlay, hide_overlay

# --- Import du système de hotkey ---
try:
    from pynput import keyboard as pynput_keyboard
    HOTKEY_BACKEND = "pynput"
except ImportError:
    try:
        import keyboard
        HOTKEY_BACKEND = "keyboard"
    except ImportError:
        print("[ERREUR] Aucun module hotkey trouvé. Installe pynput ou keyboard.")
        sys.exit(1)

# --- Essayer d'importer pystray pour l'icône system tray ---
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False


# =============================================================================
# Configuration (chargée depuis config.json)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "whisper_dictation.log")

# Configurer le logging fichier (indispensable avec pythonw.exe qui masque stdout/stderr)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
# Rediriger stderr vers le fichier log pour capturer les crashs non gérés
sys.stderr = open(LOG_FILE, "a", encoding="utf-8")

# Protéger stdout pour pythonw.exe (stdout=None → crash sur print)
if sys.stdout is None:
    sys.stdout = open(LOG_FILE, "a", encoding="utf-8")

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
VOCAB_FILE = os.path.join(BASE_DIR, "vocabulaire.txt")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "corrections.txt")
SAMPLE_RATE = 16000
CHANNELS = 1

DEFAULTS = {
    "model_size": "large-v3",
    "custom_model_path": "",
    "device": "cuda",
    "compute_type": "float16",
    "language": "fr",
    "audio_gain": 10.0,
    "auto_paste": True,
    "auto_start": False,
    "microphone": "",
    "stt_engine": "local",
    "groq_api_key": "",
    "groq_model": "whisper-large-v3-turbo",
    "fuzzy_enabled": True,
    "fuzzy_threshold": 75,
    "api_enabled": False,
    "api_host": "0.0.0.0",
    "api_port": 5000,
    "rtp_enabled": False,
    "rtp_port": 5002,
    "rtp_pool_size": 2,
    "rtp_webhook_url": "",
    "rtp_record_wav": False,
    "rtp_save_dir": "./recordings",
    "rtp_language": "fr",
}


def load_config() -> dict:
    """Charge la config depuis config.json, avec fallback sur les défauts."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        return cfg
    return dict(DEFAULTS)


config = load_config()


def load_vocabulary() -> str:
    """Charge le vocabulaire depuis vocabulaire.txt et le transforme en prompt."""
    if not os.path.exists(VOCAB_FILE):
        log(f"Fichier vocabulaire introuvable : {VOCAB_FILE}")
        return ""
    words = []
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    log(f"Vocabulaire chargé : {len(words)} mots/expressions depuis vocabulaire.txt")
    return ", ".join(words)


def load_corrections() -> list[tuple[re.Pattern, str]]:
    """Charge les corrections depuis corrections.txt. Retourne une liste de (pattern, remplacement)."""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    corrections = []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " -> " not in line:
                continue
            parts = line.split(" -> ", 1)
            erreur = parts[0].strip()
            correction = parts[1].strip()
            if erreur and correction:
                pattern = re.compile(re.escape(erreur), re.IGNORECASE)
                corrections.append((pattern, correction))
    log(f"Corrections chargées : {len(corrections)} règles depuis corrections.txt")
    return corrections


def apply_corrections(text: str) -> str:
    """Applique les corrections post-transcription."""
    corrections = load_corrections()
    if not corrections:
        return text
    original = text
    for pattern, replacement in corrections:
        text = pattern.sub(replacement, text)
    if text != original:
        log(f"Corrections appliquées : '{original}' -> '{text}'")
    return text


# =============================================================================
# État global
# =============================================================================
class AppState:
    def __init__(self):
        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.model: WhisperModel | None = None
        self.lock = threading.Lock()
        self.tray_icon = None
        self.running = True
        self.restart_requested = False
        self.ctrl_pressed = False
        self.audio_levels: collections.deque = collections.deque(maxlen=50)
        self.overlay = None
        self.settings_open = False

state = AppState()


# =============================================================================
# Utilitaire d'affichage
# =============================================================================
def log(msg: str):
    """Affiche un message horodaté et flush immédiatement."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)
    logging.info(msg)


# =============================================================================
# Gestion du cache et téléchargement du modèle
# =============================================================================

# Tailles approximatives des modèles faster-whisper (en Go)
MODEL_SIZES_GB = {
    "tiny": 0.07, "tiny.en": 0.07,
    "base": 0.14, "base.en": 0.14,
    "small": 0.46, "small.en": 0.46,
    "medium": 1.42, "medium.en": 1.42,
    "large-v1": 2.87, "large-v2": 2.87, "large-v3": 2.87,
    "large-v3-turbo": 1.62,
    "distil-large-v2": 1.51, "distil-large-v3": 1.51,
}


def _get_model_repo(model_name: str) -> str:
    """Retourne le repo HuggingFace pour un modèle faster-whisper."""
    REPO_OVERRIDES = {
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo",
        "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    }
    return REPO_OVERRIDES.get(model_name, f"Systran/faster-whisper-{model_name}")


def _get_hf_hub_dir() -> str:
    """Retourne le répertoire du cache HuggingFace Hub."""
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")


def _get_model_cache_dir(model_name: str) -> str:
    """Retourne le chemin du cache HuggingFace pour ce modèle (cherche tous les orgs)."""
    hub_dir = _get_hf_hub_dir()
    if os.path.isdir(hub_dir):
        # D'abord, chercher le repo exact (via REPO_OVERRIDES)
        repo_id = _get_model_repo(model_name)
        exact_folder = f"models--{repo_id.replace('/', '--')}"
        exact_path = os.path.join(hub_dir, exact_folder)
        if os.path.isdir(exact_path):
            return exact_path
        # Sinon, chercher tout dossier contenant le nom du modèle
        for folder in os.listdir(hub_dir):
            if folder.startswith("models--") and model_name in folder:
                return os.path.join(hub_dir, folder)
    # Fallback : construire le chemin par défaut
    repo_id = _get_model_repo(model_name)
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    return os.path.join(hub_dir, repo_folder)


def _is_model_cached(model_name: str) -> bool:
    """Vérifie si le modèle est déjà téléchargé dans le cache local."""
    # Si c'est un chemin local direct
    if os.path.isdir(model_name):
        return True
    cache_path = _get_model_cache_dir(model_name)
    snapshots_dir = os.path.join(cache_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        for d in os.listdir(snapshots_dir):
            dp = os.path.join(snapshots_dir, d)
            if os.path.isdir(dp) and len(os.listdir(dp)) > 0:
                return True
    return False


def _get_dir_size_bytes(path: str) -> int:
    """Calcule la taille totale d'un dossier en octets."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _download_model_with_progress(model_name: str, progress_window=None) -> bool:
    """Pré-télécharge le modèle avec affichage de la progression (console + fenêtre UI).

    Retourne True si le téléchargement a réussi, False sinon.
    En cas d'échec, WhisperModel() tentera lui-même le téléchargement.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("  (huggingface_hub non disponible, faster-whisper gèrera le téléchargement)")
        return False

    repo_id = _get_model_repo(model_name)
    expected_gb = MODEL_SIZES_GB.get(model_name, 1.5)
    expected_bytes = expected_gb * 1e9
    cache_path = _get_model_cache_dir(model_name)

    log(f"  Téléchargement depuis huggingface.co/{repo_id}")
    log(f"  Taille estimée : ~{expected_gb:.1f} Go")
    log(f"  Patience, cela peut prendre plusieurs minutes...")

    # Mettre à jour la fenêtre de progression
    if progress_window:
        progress_window.update_download(model_name, 0, 0.0, expected_gb)

    download_done = threading.Event()

    def monitor_progress():
        last_pct_logged = -10
        while not download_done.is_set():
            try:
                current_size = _get_dir_size_bytes(cache_path) if os.path.isdir(cache_path) else 0
                pct = min(int(current_size / expected_bytes * 100), 99) if expected_bytes > 0 else 0
                current_gb = current_size / 1e9
                # Mise à jour console (tous les 5%)
                if pct >= last_pct_logged + 5:
                    last_pct_logged = pct
                    log(f"  Progression : {pct}% ({current_gb:.2f} / {expected_gb:.1f} Go)")
                # Mise à jour fenêtre UI (à chaque cycle)
                if progress_window:
                    progress_window.update_download(model_name, pct, current_gb, expected_gb)
            except Exception:
                pass
            download_done.wait(timeout=2)

    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    try:
        snapshot_download(repo_id)
    except Exception as e:
        log(f"  ERREUR téléchargement HF Hub : {e}")
        log(f"  faster-whisper va tenter le téléchargement direct...")
        download_done.set()
        monitor_thread.join(timeout=5)
        return False
    finally:
        download_done.set()
        monitor_thread.join(timeout=5)

    final_size = _get_dir_size_bytes(cache_path) if os.path.isdir(cache_path) else 0
    log(f"  Téléchargement terminé ! ({final_size / 1e9:.2f} Go)")

    # Mise à jour finale (100%)
    if progress_window:
        progress_window.update_download(model_name, 100, final_size / 1e9, expected_gb)
    return True


# =============================================================================
# Chargement du modèle
# =============================================================================
def load_model():
    """Charge le modèle Whisper sur le device configuré (cuda/mps/cpu).

    Affiche une fenêtre de progression si un téléchargement ou un chargement
    long est nécessaire, pour que l'utilisateur sache que l'app travaille.

    Supporte un chemin personnalisé (modèle fine-tuné) via config["custom_model_path"].
    Si stt_engine="groq", aucun modèle local n'est chargé.
    """
    # Déterminer le modèle local à utiliser
    custom_path = config.get("custom_model_path", "").strip()
    if custom_path and os.path.isdir(custom_path):
        model = custom_path
        log(f"[CUSTOM] Utilisation du modèle fine-tuné : {custom_path}")
    else:
        model = config["model_size"]
        if custom_path:
            log(f"[WARN] Chemin modèle personnalisé introuvable : {custom_path}")
            log(f"       Fallback sur le modèle standard : {model}")

    is_groq = config.get("stt_engine") == "groq"

    # --- Mode Groq : charger le modèle local uniquement s'il est déjà en cache (fallback) ---
    if is_groq:
        has_local = os.path.isdir(model) or _is_model_cached(model)
        if has_local:
            log("[Groq] Moteur cloud sélectionné — chargement du modèle local en fallback...")
        else:
            log("[Groq] Moteur cloud sélectionné, pas de modèle local en cache — pas de fallback disponible.")
            state.model = None
            return

    device = config["device"]
    compute = config["compute_type"]

    progress_window = None
    needs_download = False

    # --- Vérifier si le modèle est déjà en local ---
    if os.path.isdir(model):
        log(f"[OK] Modèle local : {model}")
    elif _is_model_cached(model):
        cache_path = _get_model_cache_dir(model)
        cache_size = _get_dir_size_bytes(cache_path)
        log(f"[OK] Modèle '{model}' déjà en local ({cache_size / 1e9:.2f} Go)")
    else:
        needs_download = True
        log(f"[DOWNLOAD] Modèle '{model}' non trouvé localement.")

    # --- Créer la fenêtre de progression (téléchargement ou chargement) ---
    try:
        from download_ui import DownloadProgressWindow
        progress_window = DownloadProgressWindow()
    except Exception as e:
        log(f"(fenêtre de progression non disponible : {e})")

    try:
        # --- Télécharger si nécessaire ---
        if needs_download:
            success = _download_model_with_progress(model, progress_window=progress_window)
            if not success and progress_window:
                # Le pré-téléchargement a échoué, WhisperModel va essayer
                progress_window.update_message(
                    "⬇  Téléchargement du modèle",
                    f"{model} — téléchargement en cours..."
                )

        # --- Charger le modèle (afficher la phase de chargement) ---
        log(f"Chargement du modèle '{model}' sur {device} ({compute})...")
        if progress_window:
            progress_window.update_loading(model, device)

        try:
            state.model = WhisperModel(model, device=device, compute_type=compute)
            log(f"Modèle chargé avec succès sur {device}.")
        except Exception as e:
            log(f"ERREUR {device.upper()} ({e}), fallback sur CPU...")
            if progress_window:
                progress_window.update_loading(model, "cpu (fallback)")
            state.model = WhisperModel(model, device="cpu", compute_type="int8")
            log("Modèle chargé sur CPU (plus lent).")
    finally:
        # --- Toujours fermer la fenêtre de progression ---
        if progress_window:
            progress_window.close()

    if is_groq:
        log("[Groq] Modèle local chargé en fallback (sera utilisé si Groq échoue).")


# =============================================================================
# Enregistrement audio
# =============================================================================
def get_microphone_device() -> int | None:
    """Retourne l'index du micro configuré, ou None pour le défaut."""
    mic_name = config.get("microphone", "").strip()
    if not mic_name:
        return None
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and mic_name.lower() in d["name"].lower():
            return i
    log(f"ATTENTION : micro '{mic_name}' introuvable, utilisation du défaut.")
    return None


def audio_callback(indata, frames, time_info, status):
    """Callback appelé par sounddevice pour chaque bloc audio."""
    if status:
        log(f"Audio warning: {status}")
    if state.recording:
        state.audio_chunks.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata ** 2))) * config["audio_gain"]
        state.audio_levels.append(rms)


def start_recording():
    """Démarre l'enregistrement audio."""
    state.audio_chunks.clear()
    state.audio_levels.clear()
    state.recording = True
    try:
        mic_device = get_microphone_device()
        if mic_device is not None:
            mic_info = sd.query_devices(mic_device)
            log(f"Micro : {mic_info['name']}")
        state.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
            blocksize=1024,
            device=mic_device,
        )
        state.stream.start()
        log("=== ENREGISTREMENT EN COURS === (Ctrl+Space pour arrêter)")
        update_tray_icon(recording=True)
        state.overlay = show_overlay(state.audio_levels)
    except Exception as e:
        state.recording = False
        log(f"ERREUR micro : {e}")
        log("Vérifie que ton micro est branché et accessible.")


def stop_recording() -> np.ndarray | None:
    """Arrête l'enregistrement et retourne l'audio."""
    state.recording = False
    if state.stream:
        state.stream.stop()
        state.stream.close()
        state.stream = None

    update_tray_icon(recording=False)
    hide_overlay(state.overlay)
    state.overlay = None

    if not state.audio_chunks:
        log("Aucun audio enregistré.")
        return None

    audio = np.concatenate(state.audio_chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE

    # Amplifier l'audio (rechargé à chaque fois depuis config)
    config.update(load_config())
    gain = config["audio_gain"]
    level_before = np.abs(audio).max()
    audio = np.clip(audio * gain, -1.0, 1.0)
    level_after = np.abs(audio).max()
    log(f"=== ENREGISTREMENT STOP === ({duration:.1f}s, niveau {level_before:.3f} -> {level_after:.3f})")
    return audio


# =============================================================================
# Transcription
# =============================================================================
def _transcribe_local(audio: np.ndarray) -> str:
    """Transcrit l'audio avec faster-whisper (local)."""
    prompt = load_vocabulary()
    segments, info = state.model.transcribe(
        audio,
        language=config["language"],
        beam_size=5,
        vad_filter=False,
        initial_prompt=prompt,
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    text = " ".join(text_parts).strip()
    log(f"Langue : {info.language} (confiance: {info.language_probability:.0%})")
    return text


def _transcribe_groq(audio: np.ndarray) -> str:
    """Transcrit l'audio via l'API Groq Cloud."""
    import io
    import wave

    api_key = config.get("groq_api_key", "").strip()
    if not api_key:
        raise ValueError("Clé API Groq non configurée. Ouvrez les paramètres pour la saisir.")

    groq_model = config.get("groq_model", "whisper-large-v3-turbo")
    language = config.get("language", "fr")

    # Convertir numpy float32 → WAV bytes en mémoire
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    wav_buffer.seek(0)

    from groq import Groq
    client = Groq(api_key=api_key)

    params = {
        "file": ("audio.wav", wav_buffer, "audio/wav"),
        "model": groq_model,
        "response_format": "text",
    }
    if language and language != "auto":
        params["language"] = language

    prompt = load_vocabulary()
    if prompt:
        params["prompt"] = prompt

    transcription = client.audio.transcriptions.create(**params)

    # L'API retourne du texte brut quand response_format="text"
    text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    log(f"[Groq] Modèle: {groq_model}")
    return text


def transcribe(audio: np.ndarray) -> str:
    """Transcrit l'audio avec le moteur configuré (local ou Groq).

    Si Groq est sélectionné mais échoue (rate limit, erreur réseau, etc.),
    bascule automatiquement sur le modèle local s'il est chargé.
    """
    engine = config.get("stt_engine", "local")
    log(f"Transcription en cours ({engine})...")
    t0 = time.perf_counter()

    if engine == "groq":
        try:
            text = _transcribe_groq(audio)
        except Exception as e:
            log(f"[Groq] ERREUR : {e}")
            if state.model is not None:
                log("[Groq → Local] Fallback sur le modèle local...")
                text = _transcribe_local(audio)
            else:
                log("[Groq] Pas de modèle local disponible en fallback.")
                raise
    else:
        text = _transcribe_local(audio)

    elapsed = time.perf_counter() - t0
    log(f"Transcription terminée en {elapsed:.1f}s")

    if text:
        log(f">>> {text}")
        text = apply_corrections(text)
        # Correction fuzzy des noms propres
        if config.get("fuzzy_enabled", True):
            from fuzzy_correction import apply_fuzzy_corrections
            threshold = config.get("fuzzy_threshold", 75)
            corrected = apply_fuzzy_corrections(text, threshold)
            if corrected != text:
                log(f"[FUZZY] '{text}' -> '{corrected}'")
                text = corrected
    else:
        log("(aucun texte détecté)")

    return text


# =============================================================================
# Toggle enregistrement
# =============================================================================
def toggle_recording():
    """Appelé quand Ctrl+Space est pressé."""
    log("Ctrl+Space détecté !")
    try:
        with state.lock:
            if not state.recording:
                start_recording()
            else:
                audio = stop_recording()
                if audio is not None and len(audio) > SAMPLE_RATE * 0.3:
                    text = transcribe(audio)
                    if text:
                        pyperclip.copy(text)
                        if config["auto_paste"]:
                            time.sleep(0.1)
                            from pynput.keyboard import Controller, Key
                            kb = Controller()
                            modifier = Key.cmd if IS_MAC else Key.ctrl
                            kb.press(modifier)
                            kb.tap('v')
                            kb.release(modifier)
                            log("Texte collé automatiquement !")
                        else:
                            log("Texte copié dans le presse-papier (Ctrl+V pour coller).")
                    else:
                        log("Aucun texte détecté dans l'audio.")
                else:
                    log("Enregistrement trop court (< 0.3s), ignoré.")
    except Exception as e:
        log(f"ERREUR dans toggle_recording : {e}")
        logging.exception("toggle_recording crash")


# =============================================================================
# Hotkey avec pynput (ne nécessite PAS les droits admin)
# =============================================================================
def setup_hotkey_pynput():
    """Configure Ctrl+Space via pynput."""

    def on_press(key):
        if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
            state.ctrl_pressed = True
        elif key == pynput_keyboard.Key.space and state.ctrl_pressed:
            threading.Thread(target=toggle_recording, daemon=True).start()
        # Echap global supprimé : trop de fermetures accidentelles.
        # Pour quitter : clic droit sur l'icône tray > Quitter.

    def on_release(key):
        if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
            state.ctrl_pressed = False

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    log("Hotkey Ctrl+Space enregistré via pynput (pas besoin d'admin).")
    return listener


def setup_hotkey_keyboard():
    """Configure Ctrl+Space via le module keyboard (nécessite admin)."""
    import keyboard
    keyboard.add_hotkey("ctrl+space", lambda: threading.Thread(target=toggle_recording, daemon=True).start(), suppress=True)
    log("Hotkey Ctrl+Space enregistré via keyboard.")
    log("NOTE: Si ça ne marche pas, relance en Administrateur.")


# =============================================================================
# Icône System Tray
# =============================================================================
def _load_custom_icon(color="green"):
    """Charge une icône personnalisée depuis le dossier icons/."""
    icon_dir = os.path.join(BASE_DIR, "icons")
    # Chercher icon_green.png, icon_red.png, icon_green.ico, icon_red.ico
    for ext in ("png", "ico"):
        path = os.path.join(icon_dir, f"icon_{color}.{ext}")
        if os.path.isfile(path):
            try:
                img = Image.open(path)
                img = img.resize((64, 64), Image.LANCZOS)
                return img
            except Exception:
                pass
    # Fallback : icône unique (icon.png ou icon.ico)
    for ext in ("png", "ico"):
        path = os.path.join(icon_dir, f"icon.{ext}")
        if os.path.isfile(path):
            try:
                img = Image.open(path)
                img = img.resize((64, 64), Image.LANCZOS)
                return img
            except Exception:
                pass
    return None


def create_tray_icon(color="gray"):
    """Crée une image pour l'icône du tray (custom ou générée)."""
    # Essayer de charger une icône personnalisée
    custom = _load_custom_icon(color)
    if custom is not None:
        return custom

    # Fallback : icône générée (cercle coloré avec micro)
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if color == "red":
        fill = (220, 50, 50, 255)
    elif color == "green":
        fill = (50, 180, 50, 255)
    else:
        fill = (140, 140, 140, 255)
    draw.ellipse([4, 4, size - 4, size - 4], fill=fill)
    mx, my = size // 2, size // 2
    draw.rounded_rectangle([mx - 6, my - 14, mx + 6, my + 4], radius=4, fill="white")
    draw.arc([mx - 12, my - 8, mx + 12, my + 12], start=0, end=180, fill="white", width=2)
    draw.line([mx, my + 12, mx, my + 18], fill="white", width=2)
    draw.line([mx - 6, my + 18, mx + 6, my + 18], fill="white", width=2)
    return img


def update_tray_icon(recording: bool):
    """Met à jour l'icône du tray."""
    if not HAS_TRAY or not state.tray_icon:
        return
    try:
        color = "red" if recording else "green"
        state.tray_icon.icon = create_tray_icon(color)
        state.tray_icon.title = "Enregistrement..." if recording else "Prêt (Ctrl+Space)"
    except Exception:
        pass


def on_tray_settings(icon, item):
    """Ouvrir la fenêtre de paramètres."""
    threading.Thread(target=_open_settings, daemon=True).start()


def _open_settings():
    log("Ouverture des paramètres (sous-processus)...")
    state.settings_open = True
    try:
        python = sys.executable
        script = os.path.join(BASE_DIR, "config_ui.py")
        result = subprocess.run([python, script], cwd=BASE_DIR)
        exit_code = result.returncode
        log(f"Paramètres fermés (code retour : {exit_code})")
        # Recharger la config dans tous les cas
        config.update(load_config())
        log("Paramètres rechargés.")
        # Vérifier si un redémarrage est nécessaire (exit code 1)
        if exit_code == 1:
            log("Redémarrage demandé suite au changement de paramètres...")
            state.restart_requested = True
            state.running = False
            if state.tray_icon:
                try:
                    state.tray_icon.stop()
                except Exception:
                    pass
    except Exception as e:
        log(f"ERREUR ouverture paramètres : {e}")
        logging.exception("_open_settings crash")
    finally:
        state.settings_open = False



def on_tray_quit(icon, item):
    """Quitter depuis le tray."""
    state.running = False
    icon.stop()


def run_tray():
    """Lance l'icône system tray."""
    if not HAS_TRAY:
        return
    menu = pystray.Menu(
        pystray.MenuItem("Whisper Dictation", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Paramètres", on_tray_settings),
        pystray.MenuItem("Quitter", on_tray_quit),
    )
    state.tray_icon = pystray.Icon(
        "whisper_dictation",
        create_tray_icon("green"),
        "Whisper Dictation - Prêt (Ctrl+Space)",
        menu,
    )
    state.tray_icon.run()


# =============================================================================
# Point d'entrée
# =============================================================================
def wait_for_system_ready():
    """Attend que le GPU et l'audio soient prêts (utile au démarrage)."""
    max_attempts = 10
    delay = 3  # secondes entre chaque tentative

    # Attendre que le GPU soit disponible (si configuré)
    device = config["device"]
    if device == "cuda":
        import torch
        for attempt in range(1, max_attempts + 1):
            if torch.cuda.is_available():
                log(f"GPU CUDA disponible (tentative {attempt}/{max_attempts}).")
                break
            log(f"GPU CUDA pas encore prêt, tentative {attempt}/{max_attempts}...")
            time.sleep(delay)
        else:
            log("ATTENTION : GPU CUDA non disponible après toutes les tentatives.")
    elif device == "mps":
        import torch
        if torch.backends.mps.is_available():
            log("GPU Apple MPS (Metal) disponible.")
        else:
            log("ATTENTION : MPS non disponible, le modèle utilisera le CPU en fallback.")

    # Attendre que l'audio soit disponible
    for attempt in range(1, max_attempts + 1):
        try:
            sd.query_devices(kind='input')
            log(f"Périphérique audio disponible (tentative {attempt}/{max_attempts}).")
            return
        except Exception:
            log(f"Audio pas encore prêt, tentative {attempt}/{max_attempts}...")
            time.sleep(delay)
    log("ATTENTION : Aucun périphérique audio détecté après toutes les tentatives.")


def main():
    print("=" * 60, flush=True)
    print("  Whisper Dictation - Clone SuperWhisper", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    log(f"Python {sys.version}")
    log(f"Config : modèle={config['model_size']}, device={config['device']}, gain={config['audio_gain']}, langue={config['language']}")
    log(f"Backend hotkey : {HOTKEY_BACKEND}")
    log(f"System tray : {'oui' if HAS_TRAY else 'non (installe pystray + Pillow)'}")
    log(f"Plateforme : {platform.system()} ({platform.machine()})")

    # Avertissement si modèle distil avec langue non anglaise
    ENGLISH_ONLY_MODELS = {"distil-large-v2", "distil-large-v3"}
    if config["model_size"] in ENGLISH_ONLY_MODELS and config.get("language") != "en":
        log("=" * 60)
        log("ATTENTION : Le modèle distil ne supporte que l'anglais !")
        log("Pour le français, utilisez large-v3 ou large-v3-turbo.")
        log("=" * 60)

    # Attendre que le système soit prêt (GPU + audio)
    wait_for_system_ready()

    # Lister les périphériques audio
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        log(f"Micro par défaut : {default_input['name']}")
    except Exception as e:
        log(f"ATTENTION - Problème audio : {e}")

    print(flush=True)

    # Charger le modèle
    load_model()

    print(flush=True)

    # Démarrer l'API HTTP si activée
    if config.get("api_enabled", False):
        try:
            from api.server import start_api_server
            start_api_server(state, config, transcribe)
            log(f"API HTTP active sur {config.get('api_host', '0.0.0.0')}:{config.get('api_port', 5000)}")
        except Exception as e:
            log(f"ERREUR démarrage API : {e}")
            logging.exception("API server start failed")

    print(flush=True)

    # Lancer le tray icon
    if HAS_TRAY:
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()
        log("Icône system tray active (regarde en bas à droite).")

    # Configurer le hotkey
    if HOTKEY_BACKEND == "pynput":
        listener = setup_hotkey_pynput()
    else:
        setup_hotkey_keyboard()
        listener = None

    print(flush=True)
    log("=" * 50)
    log("PRÊT ! Appuie sur Ctrl+Space pour dicter.")
    log("Le texte sera copié dans le presse-papier.")
    log("Clic droit sur l'icône tray pour quitter.")
    log("=" * 50)
    print(flush=True)

    # Boucle principale
    try:
        while state.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        log("Arrêt en cours...")
        # Arrêter l'API HTTP
        try:
            from api.server import stop_api_server
            stop_api_server()
        except Exception:
            pass
        if state.recording:
            stop_recording()
        if listener:
            listener.stop()
        elif HOTKEY_BACKEND == "keyboard":
            import keyboard
            keyboard.unhook_all()
        if state.tray_icon:
            try:
                state.tray_icon.stop()
            except Exception:
                pass

        # --- Redémarrage automatique si demandé via les paramètres ---
        if state.restart_requested:
            log("Redémarrage de l'application...")
            time.sleep(0.5)  # Laisser le temps au cleanup
            python = sys.executable
            script = os.path.abspath(__file__)
            subprocess.Popen([python, script])
            log("Nouvelle instance lancée.")
        else:
            log("Au revoir !")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"CRASH au démarrage : {e}")
        raise
