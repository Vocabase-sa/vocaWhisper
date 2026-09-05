"""
Interface de configuration pour Whisper Dictation.
Fenêtre tkinter avec onglets Général et Vocabulaire.
"""

import json
import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import sounddevice as sd

try:
    from tokenizers import Tokenizer as HFTokenizer
    _HAS_TOKENIZER = True
except ImportError:
    _HAS_TOKENIZER = False

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# Sur Windows, définir un AppUserModelID pour que la barre des tâches
# affiche notre icône au lieu de celle de pythonw.exe
if IS_WINDOWS:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        ctypes.c_wchar_p("vocabase.vocawhisper.settings")
    )

# ---------------------------------------------------------------------------
# Thème Vocabase (vocabase.be)
# ---------------------------------------------------------------------------
VOCABASE_CORAL = "#F07654"
VOCABASE_CORAL_HOVER = "#d9613f"
VOCABASE_BLUE = "#0c2d5c"
VOCABASE_GREEN = "#10b981"
VOCABASE_RED = "#ef4444"
VOCABASE_PURPLE = "#8b5cf6"
VOCABASE_BTN_DARK = "#32373c"
VOCABASE_GRAY = "#626263"
VOCABASE_LIGHT_GRAY = "#818181"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
VOCAB_FILE = os.path.join(BASE_DIR, "vocabulaire.txt")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "corrections.txt")
NOMS_PROPRES_FILE = os.path.join(BASE_DIR, "noms_propres.txt")

# Modèles wav2vec2 utilisables en INFÉRENCE. Vérifiés un par un : chacun
# possède un vocab.json, donc une tête CTC entraînée. Les bases sans tête
# (LeBenchmark) sont volontairement absentes — elles ne transcrivent rien tant
# qu'elles n'ont pas été fine-tunées, et n'ont leur place que dans l'onglet
# Training.
WAV2VEC2_MODELS = {
    "jonatasgrosman/wav2vec2-large-xlsr-53-french": (1.26, "315M — référence, rapide"),
    "bofenghuang/asr-wav2vec2-ctc-french": (3.52, "315M — mieux entraîné en FR"),
    "jonatasgrosman/wav2vec2-xls-r-1b-french": (3.85, "1 Md — lourd sur CPU"),
    "facebook/wav2vec2-large-xlsr-53-french": (1.26, "315M — modèle d'origine"),
}

# Architectures entraînables depuis l'onglet Training. Chacune a son script, ses
# modèles de base et ses hyperparamètres usuels — un CTC de 315M ne se règle pas
# comme un seq2seq de 1,5B.
TRAIN_ARCHITECTURES = {
    "whisper": {
        "script": "train.py",
        "models": [
            "openai/whisper-large-v3",
            "bofenghuang/whisper-large-v3-french",
            "openai/whisper-large-v2",
            "openai/whisper-medium",
            "openai/whisper-small",
            "openai/whisper-base",
        ],
        "defaults": {"epochs": "3", "batch": "8", "lr": "1e-5"},
        "convertible": True,
        "hint": "seq2seq — convertible en CTranslate2",
    },
    "wav2vec2": {
        "script": "train_wav2vec2.py",
        # Les modèles déjà dotés d'une tête CTC en tête de liste : partir d'eux
        # conserve leur vocabulaire et converge en quelques époques. Les bases
        # LeBenchmark n'ont pas de tête CTC — elles demandent un entraînement
        # bien plus long, mais offrent le meilleur point de départ français.
        "models": [
            "jonatasgrosman/wav2vec2-large-xlsr-53-french",
            "bofenghuang/asr-wav2vec2-ctc-french",
            "facebook/wav2vec2-large-xlsr-53-french",
            "jonatasgrosman/wav2vec2-xls-r-1b-french",
            "LeBenchmark/wav2vec2-FR-7K-large",
            "LeBenchmark/wav2vec2-FR-14K-xlarge",
        ],
        "defaults": {"epochs": "15", "batch": "8", "lr": "1e-4"},
        "convertible": False,
        "hint": "CTC — sortie en minuscules, sans ponctuation",
    },
    "moonshine": {
        "script": "train_moonshine.py",
        "models": [
            "Cornebidouil/moonshine-tiny-fr",
            "UsefulSensors/moonshine-base",
            "UsefulSensors/moonshine-tiny",
        ],
        "defaults": {"epochs": "30", "batch": "32", "lr": "1e-4"},
        "convertible": False,
        "hint": "edge — seul le tiny existe en français",
    },
}

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
    "hotkey_primary": "Ctrl+Space",
    "hotkey_secondary": "Ctrl+F2",
    "stt_engine": "local",
    "groq_api_key": "",
    "groq_model": "whisper-large-v3-turbo",
    "groq_fallback_local": False,
    "moonshine_model": "",
    "moonshine_backend": "torch",
    "moonshine_device": "auto",
    "moonshine_fallback_local": False,
    "wav2vec2_model": "",
    "wav2vec2_device": "auto",
    "wav2vec2_fallback_local": False,
    "wav2vec2_hotwords": False,
    "wav2vec2_hotword_weight": 10.0,
    "fastconformer_model": "",
    "fastconformer_backend": "onnx",
    "fastconformer_quantization": "int8",
    "fastconformer_fallback_local": False,
    "install_mode": "full",
    "fuzzy_enabled": True,
    "fuzzy_threshold": 60,
    "restore_case": True,
    "restore_case_nouns": True,
    "restore_case_sentences": True,
    "restore_case_punctuation": True,
    "restore_case_style": "title",
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


def _get_input_devices() -> list[str]:
    """Retourne la liste des noms de périphériques d'entrée audio."""
    devices = sd.query_devices()
    names = []
    for d in devices:
        if d["max_input_channels"] > 0:
            names.append(d["name"])
    return names


# --- Détection du cache modèle (pour affichage dans les settings) ---

MODEL_SIZES_GB = {
    "tiny": 0.07, "tiny.en": 0.07,
    "base": 0.14, "base.en": 0.14,
    "small": 0.46, "small.en": 0.46,
    "medium": 1.42, "medium.en": 1.42,
    "large-v1": 2.87, "large-v2": 2.87, "large-v3": 2.87,
    "large-v3-turbo": 1.62,
    "distil-large-v2": 1.51, "distil-large-v3": 1.51,
}


# Limite de tokens pour l'initial_prompt de Whisper (n_text_ctx // 2 - 1)
WHISPER_MAX_PROMPT_TOKENS = 224

# Cache du tokenizer (chargé une seule fois)
_tokenizer_cache = {"tokenizer": None, "loaded": False}


def _load_tokenizer():
    """Charge le tokenizer Whisper depuis le cache HuggingFace (une seule fois)."""
    if _tokenizer_cache["loaded"]:
        return _tokenizer_cache["tokenizer"]
    _tokenizer_cache["loaded"] = True
    if not _HAS_TOKENIZER:
        return None
    try:
        hub_dir = _get_hf_hub_dir()
        if not os.path.isdir(hub_dir):
            return None
        for root, _, files in os.walk(hub_dir):
            if "tokenizer.json" in files:
                tok = HFTokenizer.from_file(os.path.join(root, "tokenizer.json"))
                _tokenizer_cache["tokenizer"] = tok
                return tok
    except Exception:
        pass
    return None


def _count_vocab_tokens(text: str) -> int | None:
    """Compte les tokens du vocabulaire comme Whisper le ferait."""
    tok = _load_tokenizer()
    if tok is None:
        return None
    # Reproduire le format envoyé à Whisper (mots séparés par ", ")
    lines = text.strip().splitlines()
    words = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not words:
        return 0
    prompt = ", ".join(words)
    encoded = tok.encode(prompt)
    return len(encoded.ids)


def _get_hf_hub_dir() -> str:
    """Retourne le répertoire du cache HuggingFace Hub."""
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")


def _get_model_repo(model_name: str) -> str:
    """Retourne le repo HuggingFace pour un modèle faster-whisper."""
    REPO_OVERRIDES = {
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    }
    return REPO_OVERRIDES.get(model_name, f"Systran/faster-whisper-{model_name}")


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


def _hf_repo_cache_dir(repo_id: str) -> str:
    """Chemin de cache d'un dépôt Hugging Face désigné par son identifiant complet.

    _get_model_cache_dir() ne convient pas ici : elle applique les correspondances
    propres à Whisper (préfixe Systran/faster-whisper-). Pour « org/modele », le
    dossier de cache se déduit directement du nom.
    """
    return os.path.join(_get_hf_hub_dir(), f"models--{repo_id.replace('/', '--')}")


def _is_hf_repo_cached(repo_id: str) -> bool:
    """Vérifie qu'un dépôt Hugging Face est présent et non vide dans le cache."""
    snapshots = os.path.join(_hf_repo_cache_dir(repo_id), "snapshots")
    if not os.path.isdir(snapshots):
        return False
    return any(
        os.path.isdir(os.path.join(snapshots, d)) and os.listdir(os.path.join(snapshots, d))
        for d in os.listdir(snapshots)
    )


def _is_model_cached(model_name: str) -> bool:
    """Vérifie si le modèle est déjà téléchargé dans le cache local."""
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


def _get_dir_size_gb(path: str) -> float:
    """Calcule la taille totale d'un dossier en Go."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / 1e9

# --- Démarrage automatique : Windows (Startup folder) et macOS (LaunchAgent) ---

if IS_WINDOWS:
    STARTUP_FOLDER = os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup")
    SHORTCUT_NAME = "Whisper Dictation.lnk"
    VBS_LAUNCHER = os.path.join(BASE_DIR, "run_silent.vbs")
elif IS_MAC:
    _LAUNCH_AGENT_DIR = os.path.expanduser("~/Library/LaunchAgents")
    _PLIST_NAME = "com.whisper-dictation.plist"
    _PLIST_PATH = os.path.join(_LAUNCH_AGENT_DIR, _PLIST_NAME)


def _get_shortcut_path() -> str:
    if IS_WINDOWS:
        return os.path.join(STARTUP_FOLDER, SHORTCUT_NAME)
    elif IS_MAC:
        return _PLIST_PATH
    return ""


def _startup_shortcut_exists() -> bool:
    path = _get_shortcut_path()
    return os.path.exists(path) if path else False


def _create_startup_shortcut():
    """Crée le démarrage automatique (Windows: Startup folder, macOS: LaunchAgent)."""
    if IS_WINDOWS:
        shortcut_path = _get_shortcut_path()
        ps_script = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "wscript.exe"; '
            f'$s.Arguments = """{VBS_LAUNCHER}"""; '
            f'$s.WorkingDirectory = "{BASE_DIR}"; '
            f'$s.Description = "Whisper Dictation - Démarrage automatique"; '
            f'$s.Save()'
        )
        subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
    elif IS_MAC:
        os.makedirs(_LAUNCH_AGENT_DIR, exist_ok=True)
        python_path = os.path.join(BASE_DIR, "venv", "bin", "python3")
        script_path = os.path.join(BASE_DIR, "whisper_dictation.py")
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper-dictation</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{BASE_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"""
        with open(_PLIST_PATH, "w") as f:
            f.write(plist_content)


def _remove_startup_shortcut():
    """Supprime le démarrage automatique."""
    path = _get_shortcut_path()
    if path and os.path.exists(path):
        os.remove(path)


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Compléter avec les défauts si clés manquantes
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        return cfg
    return dict(DEFAULTS)


def save_config(cfg: dict):
    print(f"[config_ui] Sauvegarde config dans : {CONFIG_FILE}", flush=True)
    print(f"[config_ui] Contenu : {cfg}", flush=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"[config_ui] Config sauvegardée !", flush=True)


def load_vocab() -> str:
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_vocab(text: str):
    print(f"[config_ui] Sauvegarde vocabulaire dans : {VOCAB_FILE}", flush=True)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[config_ui] Vocabulaire sauvegardé !", flush=True)


def load_corrections() -> str:
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_corrections(text: str):
    print(f"[config_ui] Sauvegarde corrections dans : {CORRECTIONS_FILE}", flush=True)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[config_ui] Corrections sauvegardées !", flush=True)


def load_noms_propres() -> str:
    if os.path.exists(NOMS_PROPRES_FILE):
        with open(NOMS_PROPRES_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_noms_propres(text: str):
    print(f"[config_ui] Sauvegarde noms propres dans : {NOMS_PROPRES_FILE}", flush=True)
    with open(NOMS_PROPRES_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[config_ui] Noms propres sauvegardés !", flush=True)


def _win32_set_taskbar_icon(root, ico_path):
    """Force l'icône dans la barre des tâches Windows via Win32 API."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32

        hwnd = int(root.wm_frame(), 16)

        # Charger et envoyer l'icône via WM_SETICON
        WM_SETICON = 0x0080
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        hicon = user32.LoadImageW(
            0, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE
        )
        if hicon:
            user32.SendMessageW(hwnd, WM_SETICON, 1, hicon)  # ICON_BIG
            user32.SendMessageW(hwnd, WM_SETICON, 0, hicon)  # ICON_SMALL

        # Définir l'AppUserModelID sur la fenêtre elle-même via IPropertyStore
        # C'est ce qui fait que Windows 11 utilise NOTRE icône dans la taskbar
        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", ctypes.c_ulong)]

        # PKEY_AppUserModel_ID = {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, 5
        PKEY_AppUserModel_ID = PROPERTYKEY(
            GUID(0x9F4C2855, 0x9F79, 0x4B39, (ctypes.c_ubyte * 8)(0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3)),
            5,
        )

        # PKEY_AppUserModel_RelaunchIconResource = {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, 2
        PKEY_AppUserModel_RelaunchIconResource = PROPERTYKEY(
            GUID(0x9F4C2855, 0x9F79, 0x4B39, (ctypes.c_ubyte * 8)(0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3)),
            2,
        )

        class PROPVARIANT(ctypes.Structure):
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("wReserved1", ctypes.c_ushort),
                ("wReserved2", ctypes.c_ushort),
                ("wReserved3", ctypes.c_ushort),
                ("pwszVal", ctypes.c_wchar_p),
            ]

        # IPropertyStore IID
        IID_IPropertyStore = GUID(
            0x886D8EEB, 0x8CF2, 0x4446,
            (ctypes.c_ubyte * 8)(0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xCF, 0x99),
        )

        # SHGetPropertyStoreForWindow
        SHGetPropertyStoreForWindow = shell32.SHGetPropertyStoreForWindow
        SHGetPropertyStoreForWindow.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        SHGetPropertyStoreForWindow.restype = ctypes.HRESULT

        pps = ctypes.c_void_p()
        hr = SHGetPropertyStoreForWindow(hwnd, ctypes.byref(IID_IPropertyStore), ctypes.byref(pps))
        if hr == 0 and pps.value:
            # IPropertyStore::SetValue est à l'index 6 dans la vtable
            vtable = ctypes.cast(pps, ctypes.POINTER(ctypes.c_void_p))
            vtable = ctypes.cast(vtable[0], ctypes.POINTER(ctypes.c_void_p))

            # SetValue(this, key, propvar)
            SetValue = ctypes.CFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(PROPERTYKEY),
                ctypes.POINTER(PROPVARIANT),
            )(vtable[6])

            # Définir l'AppUserModelID
            VT_LPWSTR = 31
            pv = PROPVARIANT()
            pv.vt = VT_LPWSTR
            pv.pwszVal = "vocabase.vocawhisper.settings"
            SetValue(pps, ctypes.byref(PKEY_AppUserModel_ID), ctypes.byref(pv))

            # Définir l'icône de relaunch (utilisée par la taskbar)
            pv2 = PROPVARIANT()
            pv2.vt = VT_LPWSTR
            pv2.pwszVal = ico_path + ",0"
            SetValue(pps, ctypes.byref(PKEY_AppUserModel_RelaunchIconResource), ctypes.byref(pv2))

            # Release
            Release = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
            Release(pps)
    except Exception:
        pass


class ConfigWindow:
    def __init__(self, on_close_callback=None):
        self.on_close_callback = on_close_callback
        self.cfg = load_config()

        self.root = tk.Tk()
        self.root.title("VocaWhisper - Paramètres")
        self.root.geometry("700x750")
        self.root.resizable(True, True)
        self.root.minsize(700, 700)

        # Icône de la fenêtre (Vocabase)
        self._set_window_icon()

        # Style Vocabase
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 4])
        style.configure("Accent.TButton", background=VOCABASE_CORAL, foreground="white")

        # --- Boutons en bas (packés EN PREMIER avec side=bottom pour toujours être visibles) ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(side="bottom", fill="x")

        save_btn = tk.Button(
            btn_frame,
            text=" Sauvegarder ",
            command=self._save_and_close,
            bg=VOCABASE_CORAL,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            activebackground=VOCABASE_CORAL_HOVER,
            activeforeground="white",
        )
        save_btn.pack(side="right", padx=(5, 0))

        restart_btn = tk.Button(
            btn_frame,
            text=" Redémarrer ",
            command=self._save_and_restart,
            bg=VOCABASE_BLUE,
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
            activebackground="#0a2248",
            activeforeground="white",
        )
        restart_btn.pack(side="right", padx=(5, 0))

        ttk.Button(btn_frame, text="Annuler", command=self._cancel).pack(side="right")

        # Onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # --- Onglet Général (défilable) ---
        # Cet onglet est le plus chargé : réglages du modèle, du micro, des
        # raccourcis, puis le bloc du moteur STT sélectionné. Sans défilement,
        # le bas de l'onglet sort de la fenêtre sur un écran peu haut.
        general_container = ttk.Frame(notebook)
        notebook.add(general_container, text="Général")

        general_canvas = tk.Canvas(general_container, highlightthickness=0)
        general_scroll = ttk.Scrollbar(
            general_container, orient="vertical", command=general_canvas.yview,
        )
        tab_general = ttk.Frame(general_canvas, padding=15)

        general_window = general_canvas.create_window(
            (0, 0), window=tab_general, anchor="nw",
        )
        general_canvas.configure(yscrollcommand=general_scroll.set)

        def _sync_general_scroll(_event=None):
            """Recalcule la zone défilable et fait suivre la largeur du contenu."""
            general_canvas.configure(scrollregion=general_canvas.bbox("all"))
            general_canvas.itemconfigure(general_window, width=general_canvas.winfo_width())

        tab_general.bind("<Configure>", _sync_general_scroll)
        general_canvas.bind("<Configure>", _sync_general_scroll)

        def _on_general_wheel(event):
            """Molette : ne défile que si le contenu dépasse réellement."""
            bbox = general_canvas.bbox("all")
            if bbox and bbox[3] > general_canvas.winfo_height():
                general_canvas.yview_scroll(-1 * (event.delta // 120), "units")

        general_canvas.bind_all("<MouseWheel>", _on_general_wheel)

        general_canvas.pack(side="left", fill="both", expand=True)
        general_scroll.pack(side="right", fill="y")

        row = 0

        # Modèle
        ttk.Label(tab_general, text="Modèle Whisper :").grid(row=row, column=0, sticky="w", pady=6)
        self.model_var = tk.StringVar(value=self.cfg["model_size"])
        model_combo = ttk.Combobox(tab_general, textvariable=self.model_var, state="readonly", width=18,
                                   values=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3"])
        model_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        # Indicateur cache modèle (✅ en local / ⬇ à télécharger)
        self.model_status_label = tk.Label(
            tab_general, text="", font=("Segoe UI", 9), anchor="w",
        )
        self.model_status_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 0), padx=(0, 0))
        row += 1

        # Avertissement modèle anglais uniquement
        self.model_warning_label = tk.Label(
            tab_general, text="", font=("Segoe UI", 9), anchor="w", fg=VOCABASE_RED,
        )
        self.model_warning_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4), padx=(0, 0))
        self._update_model_status()
        model_combo.bind("<<ComboboxSelected>>", lambda e: self._update_model_status())
        row += 1

        # Checkbox modèle fine-tuné
        self.use_finetuned_var = tk.BooleanVar(value=bool(self.cfg.get("custom_model_path", "").strip()))
        self.custom_model_var = tk.StringVar(value=self.cfg.get("custom_model_path", ""))
        self.use_finetuned_cb = ttk.Checkbutton(
            tab_general, text="Utiliser un modèle fine-tuné",
            variable=self.use_finetuned_var, command=self._toggle_finetuned,
        )
        self.use_finetuned_cb.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 2))
        row += 1

        # Chemin du modèle fine-tuné
        self.custom_model_frame = ttk.Frame(tab_general)
        self.custom_model_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 2), padx=(20, 0))
        self.custom_entry = ttk.Entry(self.custom_model_frame, textvariable=self.custom_model_var, width=45)
        self.custom_entry.pack(side="left", fill="x", expand=True)
        self.custom_browse_btn = tk.Button(
            self.custom_model_frame, text="...", command=self._browse_custom_model,
            font=("Segoe UI", 9), padx=6, cursor="hand2",
        )
        self.custom_browse_btn.pack(side="left", padx=(4, 0))
        row += 1

        # Indicateur modèle personnalisé
        self.custom_model_status = tk.Label(
            tab_general, text="", font=("Segoe UI", 8), anchor="w", fg=VOCABASE_GRAY,
        )
        self.custom_model_status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4), padx=(20, 0))
        self.custom_model_var.trace_add("write", lambda *_: self._update_custom_model_status())

        # Références aux widgets modèle standard pour les griser
        self.model_combo = model_combo
        # Appliquer l'état initial
        self._toggle_finetuned()
        row += 1

        # Device
        self.device_label = ttk.Label(tab_general, text="Device :")
        self.device_label.grid(row=row, column=0, sticky="w", pady=6)
        self.device_var = tk.StringVar(value=self.cfg["device"])
        device_values = ["cuda", "cpu"]
        if IS_MAC:
            device_values = ["mps", "cpu"]
        self.device_combo = ttk.Combobox(tab_general, textvariable=self.device_var, state="readonly", width=18,
                                    values=device_values)
        self.device_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        # Compute type
        self.compute_label = ttk.Label(tab_general, text="Précision :")
        self.compute_label.grid(row=row, column=0, sticky="w", pady=6)
        self.compute_var = tk.StringVar(value=self.cfg["compute_type"])
        self.compute_combo = ttk.Combobox(tab_general, textvariable=self.compute_var, state="readonly", width=18,
                                     values=["float16", "float32", "int8"])
        self.compute_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        # Langue
        ttk.Label(tab_general, text="Langue :").grid(row=row, column=0, sticky="w", pady=6)
        self.lang_var = tk.StringVar(value=self.cfg["language"] if self.cfg["language"] else "auto")
        lang_combo = ttk.Combobox(tab_general, textvariable=self.lang_var, state="readonly", width=18,
                                  values=["fr", "en", "de", "es", "nl", "it", "pt", "auto"])
        lang_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self._update_model_status())
        row += 1

        # Gain audio
        ttk.Label(tab_general, text="Gain micro :").grid(row=row, column=0, sticky="w", pady=6)
        gain_frame = ttk.Frame(tab_general)
        gain_frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=6, padx=(10, 0))
        self.gain_var = tk.DoubleVar(value=self.cfg["audio_gain"])
        self.gain_label = ttk.Label(gain_frame, text=f"x{self.cfg['audio_gain']:.1f}", width=5)
        gain_scale = ttk.Scale(gain_frame, from_=1.0, to=20.0, variable=self.gain_var, orient="horizontal", length=200,
                               command=self._update_gain_label)
        gain_scale.pack(side="left")
        self.gain_label.pack(side="left", padx=(8, 0))
        row += 1

        # Microphone
        ttk.Label(tab_general, text="Microphone :").grid(row=row, column=0, sticky="w", pady=6)
        mic_names = _get_input_devices()
        mic_values = ["(défaut système)"] + mic_names
        current_mic = self.cfg.get("microphone", "").strip()
        if current_mic:
            # Trouver la correspondance dans la liste
            matched = next((n for n in mic_names if current_mic.lower() in n.lower()), "(défaut système)")
        else:
            matched = "(défaut système)"
        self.mic_var = tk.StringVar(value=matched)
        mic_combo = ttk.Combobox(tab_general, textvariable=self.mic_var, state="readonly", width=40,
                                 values=mic_values)
        mic_combo.grid(row=row, column=1, columnspan=2, sticky="w", pady=6, padx=(10, 0))
        row += 1

        # --- Raccourcis clavier ---
        hotkey_values = [
            "Ctrl+Space", "Ctrl+²", "Ctrl+F1", "Ctrl+F2", "Ctrl+F3",
            "Ctrl+F4", "Ctrl+F5", "Ctrl+Shift+D",
            "Ctrl+Shift+A", "Ctrl+Shift+Space",
        ]

        ttk.Label(tab_general, text="Raccourci 1 :").grid(row=row, column=0, sticky="w", pady=4)
        self.hotkey1_var = tk.StringVar(value=self.cfg.get("hotkey_primary", "Ctrl+Space"))
        hk1_combo = ttk.Combobox(tab_general, textvariable=self.hotkey1_var, state="readonly", width=18,
                                 values=hotkey_values)
        hk1_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        ttk.Label(tab_general, text="Raccourci 2 :").grid(row=row, column=0, sticky="w", pady=4)
        self.hotkey2_var = tk.StringVar(value=self.cfg.get("hotkey_secondary", "Ctrl+Shift+D"))
        hk2_combo = ttk.Combobox(tab_general, textvariable=self.hotkey2_var, state="readonly", width=18,
                                 values=hotkey_values + ["Aucun"])
        hk2_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        # --- Moteur STT ---
        ttk.Separator(tab_general).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        row += 1

        ttk.Label(tab_general, text="Moteur STT :").grid(row=row, column=0, sticky="w", pady=6)
        is_groq_only = self.cfg.get("install_mode") == "groq"
        self.engine_var = tk.StringVar(value="groq" if is_groq_only else self.cfg.get("stt_engine", "local"))
        engine_values = (["groq"] if is_groq_only
                         else ["local", "groq", "moonshine", "wav2vec2", "fastconformer"])
        engine_combo = ttk.Combobox(
            tab_general, textvariable=self.engine_var,
            state="disabled" if is_groq_only else "readonly", width=18,
            values=engine_values,
        )
        engine_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        engine_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_groq_fields())
        row += 1

        # --- Paramètres du moteur sélectionné ---
        # Un seul bloc est affiché à la fois : masquer les autres plutôt que les
        # griser garde l'onglet dans la fenêtre, quel que soit le nombre de
        # moteurs disponibles.
        self.engine_frames = {}

        def engine_frame(name):
            """Crée le conteneur d'un moteur, tous sur la même ligne de grille."""
            f = ttk.Frame(tab_general)
            f.grid(row=row, column=0, columnspan=3, sticky="ew")
            self.engine_frames[name] = f
            return f

        # --- Groq ---
        fg = engine_frame("groq")
        ttk.Label(fg, text="Clé API Groq :").grid(row=0, column=0, sticky="w", pady=4)
        self.groq_key_var = tk.StringVar(value=self.cfg.get("groq_api_key", ""))
        self.groq_key_entry = ttk.Entry(fg, textvariable=self.groq_key_var, width=40, show="*")
        self.groq_key_entry.grid(row=0, column=1, columnspan=2, sticky="w", pady=4, padx=(10, 0))

        ttk.Label(fg, text="Modèle Groq :").grid(row=1, column=0, sticky="w", pady=4)
        self.groq_model_var = tk.StringVar(value=self.cfg.get("groq_model", "whisper-large-v3-turbo"))
        self.groq_model_combo = ttk.Combobox(
            fg, textvariable=self.groq_model_var, state="readonly", width=24,
            values=["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"],
        )
        self.groq_model_combo.grid(row=1, column=1, sticky="w", pady=4, padx=(10, 0))

        self.groq_fallback_var = tk.BooleanVar(value=self.cfg.get("groq_fallback_local", False))
        self.groq_fallback_check = ttk.Checkbutton(
            fg, text="Charger le modèle local en fallback (démarrage plus lent)",
            variable=self.groq_fallback_var,
        )
        self.groq_fallback_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 4))

        # --- Moonshine ---
        fm = engine_frame("moonshine")
        ttk.Label(fm, text="Modèle Moonshine :").grid(row=0, column=0, sticky="w", pady=4)
        self.moonshine_model_var = tk.StringVar(value=self.cfg.get("moonshine_model", ""))
        self.moonshine_model_entry = ttk.Entry(fm, textvariable=self.moonshine_model_var, width=40)
        self.moonshine_model_entry.grid(row=0, column=1, columnspan=2, sticky="w", pady=4, padx=(10, 0))
        ttk.Label(fm, text="Vide = Cornebidouil/moonshine-tiny-fr. Sinon : chemin du modèle fine-tuné.",
                  foreground="gray").grid(row=1, column=1, columnspan=2, sticky="w", padx=(10, 0))

        ttk.Label(fm, text="Backend :").grid(row=2, column=0, sticky="w", pady=4)
        self.moonshine_backend_var = tk.StringVar(value=self.cfg.get("moonshine_backend", "torch"))
        self.moonshine_backend_combo = ttk.Combobox(
            fm, textvariable=self.moonshine_backend_var, state="readonly", width=18,
            values=["torch", "onnx"],
        )
        self.moonshine_backend_combo.grid(row=2, column=1, sticky="w", pady=4, padx=(10, 0))

        self.moonshine_fallback_var = tk.BooleanVar(value=self.cfg.get("moonshine_fallback_local", False))
        self.moonshine_fallback_check = ttk.Checkbutton(
            fm, text="Charger Whisper en fallback (démarrage plus lent)",
            variable=self.moonshine_fallback_var,
        )
        self.moonshine_fallback_check.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))

        # --- wav2vec2 ---
        fw = engine_frame("wav2vec2")
        ttk.Label(fw, text="Modèle wav2vec2 :").grid(row=0, column=0, sticky="w", pady=4)
        self.wav2vec2_model_var = tk.StringVar(value=self.cfg.get("wav2vec2_model", ""))
        # Combobox éditable : la liste sert de raccourci, mais tout identifiant
        # Hugging Face ou chemin local reste saisissable.
        self.wav2vec2_model_entry = ttk.Combobox(
            fw, textvariable=self.wav2vec2_model_var, width=44,
            values=[""] + list(WAV2VEC2_MODELS),
        )
        self.wav2vec2_model_entry.grid(row=0, column=1, columnspan=2, sticky="w", pady=4, padx=(10, 0))
        self.wav2vec2_model_entry.bind(
            "<<ComboboxSelected>>", lambda e: self._update_wav2vec2_status()
        )
        self.wav2vec2_model_var.trace_add(
            "write", lambda *a: self._update_wav2vec2_status()
        )

        status_row = ttk.Frame(fw)
        status_row.grid(row=1, column=1, columnspan=2, sticky="w", padx=(10, 0))
        self.wav2vec2_status_label = tk.Label(
            status_row, text="", font=("Segoe UI", 8), anchor="w",
        )
        self.wav2vec2_status_label.pack(side="left")
        self.wav2vec2_dl_btn = tk.Button(
            status_row, text="Télécharger", command=self._download_wav2vec2,
            bg=VOCABASE_PURPLE, fg="white", font=("Segoe UI", 8),
            relief="flat", padx=6, cursor="hand2",
            activebackground=VOCABASE_PURPLE, activeforeground="white",
        )

        # Hotwords : boosting des noms de l'annuaire au décodage.
        self.wav2vec2_hotwords_var = tk.BooleanVar(value=self.cfg.get("wav2vec2_hotwords", False))
        ttk.Checkbutton(
            fw, text="Favoriser les noms propres au décodage (onglet « Noms propres »)",
            variable=self.wav2vec2_hotwords_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(fw, text="+27 points sur les noms propres, mais ~300 ms de plus par phrase.",
                  foreground="gray").grid(row=3, column=0, columnspan=3, sticky="w", padx=(20, 0))

        ttk.Label(fw, text="Poids :").grid(row=4, column=0, sticky="w", pady=4)
        self.wav2vec2_hotword_weight_var = tk.StringVar(
            value=str(self.cfg.get("wav2vec2_hotword_weight", 10.0))
        )
        ttk.Entry(fw, textvariable=self.wav2vec2_hotword_weight_var, width=6).grid(
            row=4, column=1, sticky="w", pady=4, padx=(10, 0))
        ttk.Label(fw, text="10 mesuré optimal ; au-delà, le décodeur force des noms à tort.",
                  foreground="gray").grid(row=5, column=0, columnspan=3, sticky="w", padx=(20, 0))

        self.wav2vec2_fallback_var = tk.BooleanVar(value=self.cfg.get("wav2vec2_fallback_local", False))
        self.wav2vec2_fallback_check = ttk.Checkbutton(
            fw, text="Charger Whisper en fallback (démarrage plus lent)",
            variable=self.wav2vec2_fallback_var,
        )
        self.wav2vec2_fallback_check.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 4))

        # --- FastConformer ---
        ff = engine_frame("fastconformer")
        ttk.Label(ff, text="Modèle FastConformer :").grid(row=0, column=0, sticky="w", pady=4)
        self.fc_model_var = tk.StringVar(value=self.cfg.get("fastconformer_model", ""))
        self.fc_model_entry = ttk.Entry(ff, textvariable=self.fc_model_var, width=40)
        self.fc_model_entry.grid(row=0, column=1, columnspan=2, sticky="w", pady=4, padx=(10, 0))
        ttk.Label(ff, text="Vide = OpenVoiceOS (ONNX). LinTO : linagora/linto_stt_fr_fastconformer + backend nemo.",
                  foreground="gray").grid(row=1, column=1, columnspan=2, sticky="w", padx=(10, 0))

        ttk.Label(ff, text="Backend :").grid(row=2, column=0, sticky="w", pady=4)
        self.fc_backend_var = tk.StringVar(value=self.cfg.get("fastconformer_backend", "onnx"))
        self.fc_backend_combo = ttk.Combobox(
            ff, textvariable=self.fc_backend_var, state="readonly", width=18,
            values=["onnx", "nemo"],
        )
        self.fc_backend_combo.grid(row=2, column=1, sticky="w", pady=4, padx=(10, 0))

        self.fc_fallback_var = tk.BooleanVar(value=self.cfg.get("fastconformer_fallback_local", False))
        self.fc_fallback_check = ttk.Checkbutton(
            ff, text="Charger Whisper en fallback (démarrage plus lent)",
            variable=self.fc_fallback_var,
        )
        self.fc_fallback_check.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))

        # "local" n'a pas de bloc : ses réglages sont ceux du haut de l'onglet.
        row += 1

        self._update_wav2vec2_status()
        self._toggle_groq_fields()

        ttk.Separator(tab_general).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 10))
        row += 1

        # Auto-paste
        self.paste_var = tk.BooleanVar(value=self.cfg["auto_paste"])
        ttk.Checkbutton(tab_general, text="Coller automatiquement après transcription",
                        variable=self.paste_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        row += 1

        # Auto-start with system
        self.autostart_var = tk.BooleanVar(value=_startup_shortcut_exists())
        if IS_WINDOWS:
            autostart_text = "Démarrer automatiquement avec Windows"
        elif IS_MAC:
            autostart_text = "Démarrer automatiquement avec macOS"
        else:
            autostart_text = "Démarrer automatiquement avec Linux"
        ttk.Checkbutton(tab_general, text=autostart_text,
                        variable=self.autostart_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        row += 1

        # --- Onglet Réseau (API HTTP) ---
        tab_network = ttk.Frame(notebook, padding=15)
        notebook.add(tab_network, text="Réseau")

        nrow = 0

        # === Section API HTTP ===
        ttk.Label(tab_network, text="API HTTP", font=("Segoe UI", 10, "bold")).grid(
            row=nrow, column=0, columnspan=3, sticky="w", pady=(0, 4))
        nrow += 1

        self.api_enabled_var = tk.BooleanVar(value=self.cfg.get("api_enabled", False))
        ttk.Checkbutton(
            tab_network, text="Activer l'API HTTP (transcription)",
            variable=self.api_enabled_var, command=self._toggle_api_fields,
        ).grid(row=nrow, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Label(tab_network, text="*redémarrage requis", foreground="gray").grid(row=nrow, column=2, padx=(5, 0))
        nrow += 1

        ttk.Label(tab_network, text="Adresse :").grid(row=nrow, column=0, sticky="w", pady=4)
        self.api_host_var = tk.StringVar(value=self.cfg.get("api_host", "0.0.0.0"))
        self.api_host_entry = ttk.Entry(tab_network, textvariable=self.api_host_var, width=18)
        self.api_host_entry.grid(row=nrow, column=1, sticky="w", pady=4, padx=(10, 0))
        nrow += 1

        ttk.Label(tab_network, text="Port :").grid(row=nrow, column=0, sticky="w", pady=4)
        self.api_port_var = tk.StringVar(value=str(self.cfg.get("api_port", 5000)))
        self.api_port_entry = ttk.Entry(tab_network, textvariable=self.api_port_var, width=8)
        self.api_port_entry.grid(row=nrow, column=1, sticky="w", pady=4, padx=(10, 0))
        nrow += 1

        self._toggle_api_fields()

        # --- Onglet Prompt initial ---
        tab_vocab = ttk.Frame(notebook, padding=15)
        notebook.add(tab_vocab, text="Prompt initial")

        ttk.Label(tab_vocab, text="Un mot ou expression par ligne (# = commentaire) :").pack(anchor="w")

        text_frame = ttk.Frame(tab_vocab)
        text_frame.pack(fill="both", expand=True, pady=(5, 0))

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.vocab_text = tk.Text(text_frame, wrap="word", font=("Consolas", 10),
                                  yscrollcommand=scrollbar.set)
        self.vocab_text.pack(fill="both", expand=True)
        scrollbar.config(command=self.vocab_text.yview)

        self.vocab_text.insert("1.0", load_vocab())

        # Compteur de tokens
        self.token_label = tk.Label(
            tab_vocab, text="", font=("Segoe UI", 9), anchor="w",
        )
        self.token_label.pack(anchor="w", pady=(4, 0))
        self._update_token_count()
        # Mettre à jour le compteur à chaque modification du texte
        self.vocab_text.bind("<<Modified>>", self._on_vocab_modified)

        # --- Onglet Corrections ---
        tab_corrections = ttk.Frame(notebook, padding=15)
        notebook.add(tab_corrections, text="Corrections")

        ttk.Label(tab_corrections, text="Format : erreur -> correction (une par ligne, # = commentaire) :").pack(anchor="w")

        corr_frame = ttk.Frame(tab_corrections)
        corr_frame.pack(fill="both", expand=True, pady=(5, 0))

        corr_scrollbar = ttk.Scrollbar(corr_frame)
        corr_scrollbar.pack(side="right", fill="y")

        self.corrections_text = tk.Text(corr_frame, wrap="word", font=("Consolas", 10),
                                        yscrollcommand=corr_scrollbar.set)
        self.corrections_text.pack(fill="both", expand=True)
        corr_scrollbar.config(command=self.corrections_text.yview)

        self.corrections_text.insert("1.0", load_corrections())

        # --- Onglet Noms propres (fuzzy matching) ---
        tab_noms = ttk.Frame(notebook, padding=15)
        notebook.add(tab_noms, text="Noms propres")

        # En-tete avec checkbox activation + seuil
        noms_header = ttk.Frame(tab_noms)
        noms_header.pack(fill="x", pady=(0, 5))

        self.fuzzy_enabled_var = tk.BooleanVar(value=self.cfg.get("fuzzy_enabled", True))
        ttk.Checkbutton(
            noms_header, text="Activer la correction fuzzy",
            variable=self.fuzzy_enabled_var,
        ).pack(side="left")

        ttk.Label(noms_header, text="Seuil :").pack(side="left", padx=(20, 4))
        self.fuzzy_threshold_var = tk.IntVar(value=self.cfg.get("fuzzy_threshold", 60))
        fuzzy_spin = ttk.Spinbox(
            noms_header, from_=50, to=100, textvariable=self.fuzzy_threshold_var, width=4,
        )
        fuzzy_spin.pack(side="left")
        ttk.Label(noms_header, text="/ 100", foreground="gray").pack(side="left", padx=(2, 0))

        # --- Remise en forme (utile pour wav2vec2 et LinTO, qui n'en produisent pas) ---
        case_frame = ttk.LabelFrame(tab_noms, text="  Remise en forme  ", padding=6)
        case_frame.pack(fill="x", pady=(6, 8))

        self.restore_case_var = tk.BooleanVar(value=self.cfg.get("restore_case", True))
        ttk.Checkbutton(
            case_frame, text="Restaurer la casse et la ponctuation",
            variable=self.restore_case_var,
            command=lambda: self._toggle_case_fields(),
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(
            case_frame,
            text="Les moteurs wav2vec2 et LinTO transcrivent en minuscules, sans ponctuation.",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=(20, 0))

        self.case_nouns_var = tk.BooleanVar(value=self.cfg.get("restore_case_nouns", True))
        self.case_nouns_check = ttk.Checkbutton(
            case_frame, text="Noms de la liste ci-dessous", variable=self.case_nouns_var,
        )
        self.case_nouns_check.grid(row=2, column=0, sticky="w", padx=(20, 0), pady=(4, 0))

        self.case_sentences_var = tk.BooleanVar(value=self.cfg.get("restore_case_sentences", True))
        self.case_sentences_check = ttk.Checkbutton(
            case_frame, text="Majuscule en début de phrase", variable=self.case_sentences_var,
        )
        self.case_sentences_check.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(4, 0))

        self.case_punct_var = tk.BooleanVar(value=self.cfg.get("restore_case_punctuation", True))
        self.case_punct_check = ttk.Checkbutton(
            case_frame, text="Point final", variable=self.case_punct_var,
        )
        self.case_punct_check.grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(4, 0))

        style_row = ttk.Frame(case_frame)
        style_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=(20, 0), pady=(6, 0))
        ttk.Label(style_row, text="Casse des noms :").pack(side="left")
        self.case_style_var = tk.StringVar(value=self.cfg.get("restore_case_style", "title"))
        self.case_style_combo = ttk.Combobox(
            style_row, textvariable=self.case_style_var, state="readonly", width=10,
            values=["title", "upper", "asis"],
        )
        self.case_style_combo.pack(side="left", padx=(6, 0))
        ttk.Label(
            style_row,
            text="title = Brisbois  ·  upper = BRISBOIS  ·  asis = comme la liste",
            foreground="gray",
        ).pack(side="left", padx=(10, 0))

        self._toggle_case_fields()

        ttk.Label(tab_noms, text="Un nom propre par ligne (noms composes acceptes, # = commentaire) :").pack(anchor="w")

        noms_frame = ttk.Frame(tab_noms)
        noms_frame.pack(fill="both", expand=True, pady=(5, 0))

        noms_scrollbar = ttk.Scrollbar(noms_frame)
        noms_scrollbar.pack(side="right", fill="y")

        self.noms_text = tk.Text(noms_frame, wrap="word", font=("Consolas", 10),
                                 yscrollcommand=noms_scrollbar.set)
        self.noms_text.pack(fill="both", expand=True)
        noms_scrollbar.config(command=self.noms_text.yview)

        self.noms_text.insert("1.0", load_noms_propres())

        # Compteur de noms
        self.noms_count_label = tk.Label(
            tab_noms, text="", font=("Segoe UI", 9), anchor="w",
        )
        self.noms_count_label.pack(anchor="w", pady=(4, 0))
        self._update_noms_count()
        self.noms_text.bind("<<Modified>>", self._on_noms_modified)

        # --- Onglet Training (Fine-tuning) ---
        tab_training = ttk.Frame(notebook, padding=10)
        notebook.add(tab_training, text="Training")
        self._training_process = None  # Subprocess en cours
        self._build_training_tab(tab_training)

        # Fermeture fenêtre avec la croix = sauvegarde aussi
        self.root.protocol("WM_DELETE_WINDOW", self._save_and_close)

    # Modèles qui ne supportent que l'anglais
    ENGLISH_ONLY_MODELS = {"distil-large-v2", "distil-large-v3"}

    def _set_window_icon(self):
        """Définit l'icône de la fenêtre et de la barre des tâches."""
        icon_dir = os.path.join(BASE_DIR, "icons")
        ico_path = None
        for name in ("icon.ico", "icon_green.ico"):
            path = os.path.join(icon_dir, name)
            if os.path.isfile(path):
                ico_path = os.path.abspath(path)
                break
        if ico_path is None:
            return
        try:
            self.root.iconbitmap(ico_path)
        except Exception:
            pass
        # Sur Windows, forcer l'icône dans la barre des tâches via Win32 API
        if IS_WINDOWS:
            self.root.update_idletasks()
            _win32_set_taskbar_icon(self.root, ico_path)

    def _browse_custom_model(self):
        """Ouvre un dialogue pour sélectionner le dossier du modèle fine-tuné."""
        from tkinter import filedialog
        path = filedialog.askdirectory(
            title="Sélectionner le dossier du modèle fine-tuné (CTranslate2)",
            initialdir=os.path.join(BASE_DIR, "fine_tuning"),
        )
        if path:
            self.custom_model_var.set(path)

    # =================================================================
    # Onglet Training
    # =================================================================
    def _build_training_tab(self, parent):
        """Construit l'interface de l'onglet Training."""
        FINE_TUNING_DIR = os.path.join(BASE_DIR, "fine_tuning")
        DATA_DIR = os.path.join(FINE_TUNING_DIR, "data")

        # --- Section 1 : Données ---
        sec_data = ttk.LabelFrame(parent, text="  1. Données  ", padding=8)
        sec_data.pack(fill="x", pady=(0, 6))

        # CSV
        row_csv = ttk.Frame(sec_data)
        row_csv.pack(fill="x", pady=2)
        ttk.Label(row_csv, text="Fichier CSV :", width=14).pack(side="left")
        self.train_csv_var = tk.StringVar(value=os.path.join(DATA_DIR, "transcriptions.csv"))
        ttk.Entry(row_csv, textvariable=self.train_csv_var, width=40).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Button(row_csv, text="...", command=lambda: self._browse_file(
            self.train_csv_var, "CSV", [("CSV", "*.csv"), ("Tous", "*.*")],
        ), font=("Segoe UI", 8), padx=4, cursor="hand2").pack(side="left", padx=(4, 0))

        # Audio dir
        row_audio = ttk.Frame(sec_data)
        row_audio.pack(fill="x", pady=2)
        ttk.Label(row_audio, text="Dossier audio :", width=14).pack(side="left")
        self.train_audio_var = tk.StringVar(value=os.path.join(DATA_DIR, "audio"))
        ttk.Entry(row_audio, textvariable=self.train_audio_var, width=40).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Button(row_audio, text="...", command=lambda: self._browse_dir(
            self.train_audio_var, "Dossier audio",
        ), font=("Segoe UI", 8), padx=4, cursor="hand2").pack(side="left", padx=(4, 0))

        # Test split + bouton Préparer
        row_prep = ttk.Frame(sec_data)
        row_prep.pack(fill="x", pady=(4, 0))
        ttk.Label(row_prep, text="Split test :", width=14).pack(side="left")
        self.train_test_size_var = tk.StringVar(value="0.1")
        ttk.Entry(row_prep, textvariable=self.train_test_size_var, width=6).pack(side="left", padx=(4, 0))
        ttk.Label(row_prep, text="(0.0 - 0.5)").pack(side="left", padx=(4, 0))
        tk.Button(
            row_prep, text="Préparer le dataset", command=self._run_prepare,
            bg=VOCABASE_BLUE, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground="#0a2248", activeforeground="white",
        ).pack(side="right")

        # --- Section 2 : Entraînement ---
        sec_train = ttk.LabelFrame(parent, text="  2. Entraînement  ", padding=8)
        sec_train.pack(fill="x", pady=(0, 6))

        # Ligne 0 : Architecture — commande le script lancé et les modèles proposés
        row_arch = ttk.Frame(sec_train)
        row_arch.pack(fill="x", pady=2)
        ttk.Label(row_arch, text="Architecture :", width=14).pack(side="left")
        self.train_arch_var = tk.StringVar(value="whisper")
        arch_combo = ttk.Combobox(
            row_arch, textvariable=self.train_arch_var, state="readonly", width=16,
            values=list(TRAIN_ARCHITECTURES),
        )
        arch_combo.pack(side="left", padx=(4, 0))
        self.train_arch_hint = ttk.Label(row_arch, text="", foreground="gray")
        self.train_arch_hint.pack(side="left", padx=(10, 0))
        arch_combo.bind("<<ComboboxSelected>>", lambda e: self._on_train_arch_change())

        # Ligne 1 : Modèle de base
        row_base = ttk.Frame(sec_train)
        row_base.pack(fill="x", pady=2)
        ttk.Label(row_base, text="Modèle de base :", width=14).pack(side="left")
        self.train_base_model_var = tk.StringVar(
            value=TRAIN_ARCHITECTURES["whisper"]["models"][0]
        )
        self.train_base_combo = ttk.Combobox(
            row_base, textvariable=self.train_base_model_var, width=38,
            values=TRAIN_ARCHITECTURES["whisper"]["models"],
        )
        self.train_base_combo.pack(side="left", padx=(4, 0))

        # Ligne 2 : Époques, Batch, LR
        row_params = ttk.Frame(sec_train)
        row_params.pack(fill="x", pady=2)
        ttk.Label(row_params, text="Époques :", width=14).pack(side="left")
        self.train_epochs_var = tk.StringVar(value="3")
        ttk.Entry(row_params, textvariable=self.train_epochs_var, width=4).pack(side="left", padx=(4, 0))
        ttk.Label(row_params, text="  Batch :").pack(side="left", padx=(8, 0))
        self.train_batch_var = tk.StringVar(value="8")
        ttk.Entry(row_params, textvariable=self.train_batch_var, width=4).pack(side="left", padx=(4, 0))
        ttk.Label(row_params, text="  LR :").pack(side="left", padx=(8, 0))
        self.train_lr_var = tk.StringVar(value="1e-5")
        ttk.Entry(row_params, textvariable=self.train_lr_var, width=8).pack(side="left", padx=(4, 0))

        # Bouton Lancer
        row_launch = ttk.Frame(sec_train)
        row_launch.pack(fill="x", pady=(4, 0))
        tk.Button(
            row_launch, text="Lancer l'entraînement", command=self._run_train,
            bg=VOCABASE_CORAL, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground=VOCABASE_CORAL_HOVER, activeforeground="white",
        ).pack(side="right")

        # --- Section 3 : Conversion CTranslate2 ---
        # Ne concerne que Whisper : CTranslate2 ne sait pas convertir un CTC ni
        # un Moonshine. La section est masquée pour les autres architectures.
        self.sec_convert = ttk.LabelFrame(parent, text="  3. Conversion CTranslate2  ", padding=8)
        sec_convert = self.sec_convert
        sec_convert.pack(fill="x", pady=(0, 6))

        row_conv = ttk.Frame(sec_convert)
        row_conv.pack(fill="x", pady=2)
        ttk.Label(row_conv, text="Quantization :", width=14).pack(side="left")
        self.train_quant_var = tk.StringVar(value="float16")
        ttk.Combobox(row_conv, textvariable=self.train_quant_var, state="readonly", width=16, values=[
            "float16", "float32", "int8", "int8_float16",
        ]).pack(side="left", padx=(4, 0))
        tk.Button(
            row_conv, text="Convertir", command=self._run_convert,
            bg=VOCABASE_PURPLE, fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground="#7c3aed", activeforeground="white",
        ).pack(side="right")

        # --- Zone de log ---
        log_frame = ttk.LabelFrame(parent, text="  Journal  ", padding=4)
        log_frame.pack(fill="both", expand=True, pady=(0, 0))

        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_inner)
        log_scroll.pack(side="right", fill="y")

        self.train_log = tk.Text(
            log_inner, wrap="word", font=("Consolas", 9), height=8,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
            yscrollcommand=log_scroll.set, state="disabled",
        )
        self.train_log.pack(fill="both", expand=True)
        log_scroll.config(command=self.train_log.yview)

        # Appliquer l'architecture initiale (renseigne l'indication et
        # affiche ou masque la section CTranslate2). Appelé ici, car cette
        # section doit déjà exister.
        self._on_train_arch_change()

        # Bouton Arrêter (caché par défaut)
        self._stop_frame = ttk.Frame(log_frame)
        self._stop_frame.pack(fill="x", pady=(4, 0))
        self._stop_btn = tk.Button(
            self._stop_frame, text="Arrêter le processus", command=self._stop_training_process,
            bg=VOCABASE_RED, fg="white", font=("Segoe UI", 9),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground="#dc2626", activeforeground="white",
        )
        # Pas de pack ici — affiché dynamiquement

    def _log_training(self, text: str):
        """Ajoute du texte dans le journal de l'onglet Training."""
        self.train_log.config(state="normal")
        self.train_log.insert("end", text)
        self.train_log.see("end")
        self.train_log.config(state="disabled")

    def _browse_file(self, var: tk.StringVar, title: str, filetypes: list):
        """Ouvre un dialogue pour sélectionner un fichier."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=title,
            initialdir=os.path.join(BASE_DIR, "fine_tuning", "data"),
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar, title: str):
        """Ouvre un dialogue pour sélectionner un dossier."""
        from tkinter import filedialog
        path = filedialog.askdirectory(
            title=title,
            initialdir=os.path.join(BASE_DIR, "fine_tuning", "data"),
        )
        if path:
            var.set(path)

    def _install_deps_if_needed(self) -> bool:
        """Vérifie et installe les dépendances fine-tuning si nécessaire.

        Retourne True si tout est OK, False si l'installation a échoué.
        """
        python = sys.executable
        # Vérifier rapidement si les packages clés sont présents
        check = subprocess.run(
            [python, "-c", "import transformers, datasets, evaluate, accelerate, ctranslate2"],
            capture_output=True, text=True,
        )
        if check.returncode == 0:
            return True  # Tout est déjà installé

        # Installation nécessaire
        self.root.after(0, self._log_training, "  Installation des dépendances fine-tuning...\n")
        req_file = os.path.join(BASE_DIR, "fine_tuning", "requirements.txt")
        proc = subprocess.Popen(
            [python, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            self.root.after(0, self._log_training, f"  {line}")
        proc.wait()
        if proc.returncode == 0:
            self.root.after(0, self._log_training, "  [OK] Dépendances installées.\n\n")
            return True
        else:
            self.root.after(0, self._log_training, "  [ERREUR] Installation des dépendances échouée.\n")
            return False

    def _run_subprocess(self, cmd: list[str], label: str):
        """Lance un sous-processus et redirige sa sortie vers le journal."""
        if self._training_process is not None and self._training_process.poll() is None:
            messagebox.showwarning("Processus en cours", "Un processus est déjà en cours d'exécution.")
            return

        self._log_training(f"\n{'='*50}\n  {label}\n{'='*50}\n")
        self._log_training(f"  > {' '.join(cmd)}\n\n")

        # Afficher le bouton Arrêter
        self._stop_btn.pack(side="right")

        def _run():
            try:
                # Installer les dépendances si nécessaire
                if not self._install_deps_if_needed():
                    self.root.after(0, lambda: self._stop_btn.pack_forget())
                    return
                self._training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=BASE_DIR,
                    bufsize=1,
                )
                for line in self._training_process.stdout:
                    # Thread-safe update via after()
                    self.root.after(0, self._log_training, line)
                self._training_process.wait()
                exit_code = self._training_process.returncode
                if exit_code == 0:
                    self.root.after(0, self._log_training, f"\n  [OK] {label} terminé avec succès !\n")
                else:
                    self.root.after(0, self._log_training, f"\n  [ERREUR] {label} échoué (code {exit_code})\n")
            except Exception as e:
                self.root.after(0, self._log_training, f"\n  [ERREUR] {e}\n")
            finally:
                self._training_process = None
                self.root.after(0, lambda: self._stop_btn.pack_forget())

        threading.Thread(target=_run, daemon=True).start()

    def _stop_training_process(self):
        """Arrête le processus en cours."""
        if self._training_process and self._training_process.poll() is None:
            self._training_process.terminate()
            self._log_training("\n  [STOP] Processus arrêté par l'utilisateur.\n")

    def _run_prepare(self):
        """Lance la préparation du dataset."""
        python = sys.executable
        script = os.path.join(BASE_DIR, "fine_tuning", "prepare_dataset.py")
        cmd = [
            python, script,
            "--csv", self.train_csv_var.get(),
            "--audio_dir", self.train_audio_var.get(),
            "--test_size", self.train_test_size_var.get(),
        ]
        self._run_subprocess(cmd, "Préparation du dataset")

    def _toggle_case_fields(self):
        """Active les options fines seulement si la remise en forme est activée."""
        state = "normal" if self.restore_case_var.get() else "disabled"
        for check in (self.case_nouns_check, self.case_sentences_check,
                      self.case_punct_check):
            check.config(state=state)
        self.case_style_combo.config(state="readonly" if state == "normal" else "disabled")

    def _update_wav2vec2_status(self):
        """Indique si le modèle wav2vec2 est en local, et propose son téléchargement."""
        # La variable est tracée dès sa création, donc avant que le bouton
        # existe : ignorer les appels trop précoces.
        if not hasattr(self, "wav2vec2_dl_btn"):
            return

        model = self.wav2vec2_model_var.get().strip() or list(WAV2VEC2_MODELS)[0]

        # Un chemin local (modèle fine-tuné) est par définition déjà présent
        if os.path.isdir(model) or os.path.isdir(os.path.join(BASE_DIR, model)):
            self.wav2vec2_status_label.config(
                text="  ✅  Modèle local", fg=VOCABASE_GREEN,
            )
            self.wav2vec2_dl_btn.pack_forget()
            return

        size = (_get_dir_size_gb(_hf_repo_cache_dir(model))
                if _is_hf_repo_cached(model) else 0.0)

        # Un dossier de cache peut exister sans les poids : un téléchargement
        # interrompu n'y laisse que les fichiers de configuration. Sous ce
        # seuil, le modèle est présent mais inutilisable.
        if 0 < size < 0.05:
            self.wav2vec2_status_label.config(
                text=f"  ⚠  Téléchargement incomplet ({size * 1000:.0f} Mo)",
                fg=VOCABASE_RED,
            )
            self.wav2vec2_dl_btn.pack(side="left", padx=(8, 0))
        elif size:
            self.wav2vec2_status_label.config(
                text=f"  ✅  En local ({size:.2f} Go)", fg=VOCABASE_GREEN,
            )
            self.wav2vec2_dl_btn.pack_forget()
        else:
            expected = WAV2VEC2_MODELS.get(model, (0, ""))[0]
            suffix = f" (~{expected:.1f} Go)" if expected else ""
            self.wav2vec2_status_label.config(
                text=f"  ⬇  À télécharger{suffix}", fg=VOCABASE_RED,
            )
            self.wav2vec2_dl_btn.pack(side="left", padx=(8, 0))

    def _download_wav2vec2(self):
        """Télécharge le modèle wav2vec2 sélectionné dans le cache Hugging Face."""
        model = self.wav2vec2_model_var.get().strip() or list(WAV2VEC2_MODELS)[0]
        expected = WAV2VEC2_MODELS.get(model, (0, ""))[0]

        from tkinter import messagebox
        if not messagebox.askyesno(
            "Téléchargement",
            f"Télécharger {model} ?\n\n"
            f"Taille approximative : {expected:.1f} Go\n\n"
            "Le téléchargement s'affiche dans le journal de l'onglet Training.",
        ):
            return

        # allow_patterns écarte les poids TensorFlow/Flax et le modèle de langage
        # n-gram : inutiles ici, et parfois plus volumineux que le modèle lui-même.
        code = (
            "from huggingface_hub import snapshot_download; "
            f"p = snapshot_download({model!r}, allow_patterns="
            "['*.json','*.txt','*.safetensors','*.bin','*.model']); "
            "print('Modele telecharge dans :', p)"
        )
        self._run_subprocess([sys.executable, "-c", code], f"Téléchargement {model}")

    def _on_train_arch_change(self):
        """Adapte les modèles proposés et les hyperparamètres à l'architecture.

        Les valeurs par défaut diffèrent fortement d'une famille à l'autre : un
        CTC de 315M se fine-tune à 1e-4 sur une quinzaine d'époques, là où un
        Whisper de 1,5B demande 1e-5 sur trois. Proposer les mêmes réglages
        partout mènerait à des entraînements dégénérés.
        """
        arch = self.train_arch_var.get()
        spec = TRAIN_ARCHITECTURES.get(arch)
        if not spec:
            return

        self.train_base_combo.config(values=spec["models"])
        self.train_base_model_var.set(spec["models"][0])

        self.train_epochs_var.set(spec["defaults"]["epochs"])
        self.train_batch_var.set(spec["defaults"]["batch"])
        self.train_lr_var.set(spec["defaults"]["lr"])

        self.train_arch_hint.config(text=spec["hint"])

        # CTranslate2 ne convertit que Whisper
        if spec["convertible"]:
            self.sec_convert.pack(fill="x", pady=(0, 6))
        else:
            self.sec_convert.pack_forget()

    def _run_train(self):
        """Lance l'entraînement après vérification GPU."""
        python = sys.executable

        # Vérifier la présence d'un GPU (CUDA ou MPS)
        gpu_check = subprocess.run(
            [python, "-c",
             "import torch; "
             "ok = torch.cuda.is_available() or (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()); "
             "print('cuda' if torch.cuda.is_available() else ('mps' if ok else 'none'))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        gpu = gpu_check.stdout.strip() if gpu_check.returncode == 0 else "none"

        if gpu == "none":
            from tkinter import messagebox
            messagebox.showwarning(
                "GPU requis",
                "Aucun GPU détecté (CUDA ou Apple MPS).\n\n"
                "Le fine-tuning nécessite un GPU NVIDIA (CUDA) "
                "ou Apple Silicon (MPS).\n\n"
                "L'entraînement sur CPU n'est pas supporté."
            )
            return

        arch = self.train_arch_var.get()
        spec = TRAIN_ARCHITECTURES.get(arch, TRAIN_ARCHITECTURES["whisper"])

        script = os.path.join(BASE_DIR, "fine_tuning", spec["script"])
        if not os.path.isfile(script):
            from tkinter import messagebox
            messagebox.showerror(
                "Script introuvable",
                f"Le script d'entraînement est absent :\n{script}",
            )
            return

        cmd = [
            python, script,
            "--base_model", self.train_base_model_var.get(),
            "--epochs", self.train_epochs_var.get(),
            "--batch_size", self.train_batch_var.get(),
            "--learning_rate", self.train_lr_var.get(),
        ]
        self._run_subprocess(cmd, f"Fine-tuning {arch}")

    def _run_convert(self):
        """Lance la conversion CTranslate2."""
        python = sys.executable
        script = os.path.join(BASE_DIR, "fine_tuning", "convert_to_ct2.py")
        cmd = [
            python, script,
            "--quantization", self.train_quant_var.get(),
        ]
        self._run_subprocess(cmd, "Conversion CTranslate2")

    def _toggle_finetuned(self):
        """Active/désactive le modèle fine-tuné et grise la combobox standard en conséquence."""
        use_ft = self.use_finetuned_var.get()
        if use_ft:
            # Griser le modèle standard
            self.model_combo.config(state="disabled")
            self.custom_entry.config(state="normal")
            self.custom_browse_btn.config(state="normal")
            self._update_custom_model_status()
        else:
            # Réactiver le modèle standard, désactiver le chemin fine-tuné
            self.model_combo.config(state="readonly")
            self.custom_entry.config(state="disabled")
            self.custom_browse_btn.config(state="disabled")
            self.custom_model_status.config(text="", fg=VOCABASE_GRAY)

    def _update_custom_model_status(self):
        """Met à jour l'indicateur du modèle personnalisé."""
        if not self.use_finetuned_var.get():
            self.custom_model_status.config(text="", fg=VOCABASE_GRAY)
            return
        path = self.custom_model_var.get().strip()
        if not path:
            self.custom_model_status.config(
                text="  Sélectionnez le dossier du modèle CTranslate2",
                fg=VOCABASE_GRAY,
            )
        elif os.path.isdir(path):
            # Vérifier si c'est un modèle CTranslate2 valide
            model_bin = os.path.join(path, "model.bin")
            if os.path.isfile(model_bin):
                size_gb = os.path.getsize(model_bin) / 1e9
                self.custom_model_status.config(
                    text=f"  \u2705  Modèle CTranslate2 trouvé ({size_gb:.2f} Go) — *redémarrage requis",
                    fg=VOCABASE_GREEN,
                )
            else:
                self.custom_model_status.config(
                    text="  \u26a0  Dossier trouvé mais pas de model.bin",
                    fg="#e67e22",
                )
        else:
            self.custom_model_status.config(
                text=f"  \u274c  Dossier introuvable : {path}",
                fg=VOCABASE_RED,
            )

    def _toggle_groq_fields(self):
        """N'affiche que le bloc du moteur sélectionné et ajuste le mode groq-only."""
        engine = self.engine_var.get()
        is_groq_only = self.cfg.get("install_mode") == "groq"

        # Un seul bloc visible : les autres sont retirés de la grille, pas
        # seulement grisés, pour que l'onglet tienne dans la fenêtre.
        for name, frame in self.engine_frames.items():
            if name == engine:
                frame.grid()
            else:
                frame.grid_remove()

        # En mode groq-only, aucun modèle local n'est installé : pas de fallback.
        if is_groq_only:
            for var in (self.groq_fallback_var, self.moonshine_fallback_var,
                        self.wav2vec2_fallback_var, self.fc_fallback_var):
                var.set(False)
            for check in (self.groq_fallback_check, self.moonshine_fallback_check,
                          self.wav2vec2_fallback_check, self.fc_fallback_check):
                check.config(state="disabled")

        # Widgets modèle local : désactivés en mode groq-only
        if is_groq_only:
            local_state = "disabled"
            self.model_combo.config(state="disabled")
            self.device_combo.config(state="disabled")
            self.compute_combo.config(state="disabled")
            self.use_finetuned_cb.config(state="disabled")
            self.custom_entry.config(state="disabled")
            self.custom_browse_btn.config(state="disabled")
            # Forcer le moteur sur groq (pas de choix local possible)
            self.engine_var.set("groq")
        else:
            # Rétablir l'état normal des widgets locaux
            if not self.use_finetuned_var.get():
                self.model_combo.config(state="readonly")
            self.device_combo.config(state="readonly")
            self.compute_combo.config(state="readonly")
            self.use_finetuned_cb.config(state="normal")

    def _toggle_api_fields(self):
        """Active/désactive les champs API selon la checkbox."""
        enabled = self.api_enabled_var.get()
        state = "normal" if enabled else "disabled"
        self.api_host_entry.config(state=state)
        self.api_port_entry.config(state=state)

    def _update_model_status(self):
        """Met à jour le label indiquant si le modèle est en local ou à télécharger."""
        model = self.model_var.get()
        if _is_model_cached(model):
            cache_path = _get_model_cache_dir(model)
            size_gb = _get_dir_size_gb(cache_path)
            self.model_status_label.config(
                text=f"  \u2705  En local ({size_gb:.2f} Go)",
                fg=VOCABASE_GREEN,
            )
        else:
            expected_gb = MODEL_SIZES_GB.get(model, 0)
            if expected_gb > 0:
                self.model_status_label.config(
                    text=f"  \u2b07  \u00c0 t\u00e9l\u00e9charger (~{expected_gb:.1f} Go)",
                    fg=VOCABASE_RED,
                )
            else:
                self.model_status_label.config(
                    text=f"  \u2b07  \u00c0 t\u00e9l\u00e9charger",
                    fg=VOCABASE_RED,
                )

        # Avertissement si modèle anglais uniquement + langue non anglaise
        lang = self.lang_var.get() if hasattr(self, "lang_var") else "fr"
        if model in self.ENGLISH_ONLY_MODELS and lang != "en":
            self.model_warning_label.config(
                text=f"  \u26a0  Ce mod\u00e8le ne supporte que l'anglais ! Utilisez large-v3 ou large-v3-turbo pour le fran\u00e7ais.",
            )
        else:
            self.model_warning_label.config(text="")

    def _on_vocab_modified(self, _=None):
        """Appelé quand le texte du vocabulaire change."""
        if self.vocab_text.edit_modified():
            self._update_token_count()
            self.vocab_text.edit_modified(False)

    def _update_token_count(self):
        """Met à jour le compteur de tokens du vocabulaire."""
        text = self.vocab_text.get("1.0", "end-1c")
        token_count = _count_vocab_tokens(text)
        max_tokens = WHISPER_MAX_PROMPT_TOKENS
        if token_count is not None:
            remaining = max_tokens - token_count
            if remaining >= 0:
                self.token_label.config(
                    text=f"Tokens : {token_count} / {max_tokens} utilises   |   {remaining} restants",
                    fg=VOCABASE_GREEN,
                )
            else:
                self.token_label.config(
                    text=f"Tokens : {token_count} / {max_tokens} utilises   |   {abs(remaining)} en trop (sera tronque)",
                    fg=VOCABASE_RED,
                )
        else:
            # Pas de tokenizer disponible : estimation approximative
            lines = text.strip().splitlines()
            words = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
            approx = len(", ".join(words).split()) if words else 0
            self.token_label.config(
                text=f"~{approx} mots (tokenizer non disponible, estimation)",
                fg=VOCABASE_GRAY,
            )

    def _on_noms_modified(self, _=None):
        """Appele quand le texte des noms propres change."""
        if self.noms_text.edit_modified():
            self._update_noms_count()
            self.noms_text.edit_modified(False)

    def _update_noms_count(self):
        """Met a jour le compteur de noms propres."""
        text = self.noms_text.get("1.0", "end-1c")
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
        count = len(lines)
        self.noms_count_label.config(
            text=f"{count} nom(s) propre(s) enregistre(s)",
            fg=VOCABASE_GREEN if count > 0 else VOCABASE_GRAY,
        )

    def _update_gain_label(self, _=None):
        self.gain_label.config(text=f"x{self.gain_var.get():.1f}")

    def _save_and_close(self):
        try:
            # Détecter si modèle/device/compute a changé
            needs_restart = (
                self.model_var.get() != self.cfg["model_size"]
                or self.device_var.get() != self.cfg["device"]
                or self.compute_var.get() != self.cfg["compute_type"]
                or (self.custom_model_var.get().strip() if self.use_finetuned_var.get() else "") != self.cfg.get("custom_model_path", "")
                or self.hotkey1_var.get() != self.cfg.get("hotkey_primary", "Ctrl+Space")
                or self.hotkey2_var.get() != self.cfg.get("hotkey_secondary", "Ctrl+Shift+D")
                or self.engine_var.get() != self.cfg.get("stt_engine", "local")
                or self.groq_fallback_var.get() != self.cfg.get("groq_fallback_local", False)
                or self.moonshine_model_var.get().strip() != self.cfg.get("moonshine_model", "")
                or self.moonshine_backend_var.get() != self.cfg.get("moonshine_backend", "torch")
                or self.moonshine_fallback_var.get() != self.cfg.get("moonshine_fallback_local", False)
                or self.wav2vec2_model_var.get().strip() != self.cfg.get("wav2vec2_model", "")
                or self.wav2vec2_fallback_var.get() != self.cfg.get("wav2vec2_fallback_local", False)
                or self.wav2vec2_hotwords_var.get() != self.cfg.get("wav2vec2_hotwords", False)
                or self.fc_model_var.get().strip() != self.cfg.get("fastconformer_model", "")
                or self.fc_backend_var.get() != self.cfg.get("fastconformer_backend", "onnx")
                or self.fc_fallback_var.get() != self.cfg.get("fastconformer_fallback_local", False)
                or self.api_enabled_var.get() != self.cfg.get("api_enabled", False)
                or self.api_host_var.get().strip() != self.cfg.get("api_host", "0.0.0.0")
                or int(self.api_port_var.get()) != self.cfg.get("api_port", 5000)
            )

            # Sauvegarder config
            lang = self.lang_var.get()
            auto_start = self.autostart_var.get()
            # Microphone : vide si "(défaut système)"
            mic_selection = self.mic_var.get()
            mic_value = "" if mic_selection == "(défaut système)" else mic_selection

            new_cfg = {
                "model_size": self.model_var.get(),
                "custom_model_path": self.custom_model_var.get().strip() if self.use_finetuned_var.get() else "",
                "device": self.device_var.get(),
                "compute_type": self.compute_var.get(),
                "language": lang if lang != "auto" else None,
                "audio_gain": round(self.gain_var.get(), 1),
                "auto_paste": self.paste_var.get(),
                "auto_start": auto_start,
                "microphone": mic_value,
                "hotkey_primary": self.hotkey1_var.get(),
                "hotkey_secondary": self.hotkey2_var.get(),
                "stt_engine": self.engine_var.get(),
                "groq_api_key": self.groq_key_var.get().strip(),
                "groq_model": self.groq_model_var.get(),
                "groq_fallback_local": self.groq_fallback_var.get(),
                "moonshine_model": self.moonshine_model_var.get().strip(),
                "moonshine_backend": self.moonshine_backend_var.get(),
                "moonshine_fallback_local": self.moonshine_fallback_var.get(),
                "wav2vec2_model": self.wav2vec2_model_var.get().strip(),
                "wav2vec2_fallback_local": self.wav2vec2_fallback_var.get(),
                "wav2vec2_hotwords": self.wav2vec2_hotwords_var.get(),
                "wav2vec2_hotword_weight": float(self.wav2vec2_hotword_weight_var.get() or 10.0),
                "fastconformer_model": self.fc_model_var.get().strip(),
                "fastconformer_backend": self.fc_backend_var.get(),
                "fastconformer_fallback_local": self.fc_fallback_var.get(),
                "fuzzy_enabled": self.fuzzy_enabled_var.get(),
                "fuzzy_threshold": self.fuzzy_threshold_var.get(),
                "restore_case": self.restore_case_var.get(),
                "restore_case_nouns": self.case_nouns_var.get(),
                "restore_case_sentences": self.case_sentences_var.get(),
                "restore_case_punctuation": self.case_punct_var.get(),
                "restore_case_style": self.case_style_var.get(),
                "api_enabled": self.api_enabled_var.get(),
                "api_host": self.api_host_var.get().strip(),
                "api_port": int(self.api_port_var.get()),
            }
            save_config(new_cfg)

            # Gérer le raccourci Startup Windows
            if auto_start and not _startup_shortcut_exists():
                _create_startup_shortcut()
                print("[config_ui] Raccourci Startup créé.", flush=True)
            elif not auto_start and _startup_shortcut_exists():
                _remove_startup_shortcut()
                print("[config_ui] Raccourci Startup supprimé.", flush=True)

            # Sauvegarder vocabulaire
            vocab_content = self.vocab_text.get("1.0", "end-1c")
            save_vocab(vocab_content)

            # Sauvegarder corrections
            corr_content = self.corrections_text.get("1.0", "end-1c")
            save_corrections(corr_content)

            # Sauvegarder noms propres
            noms_content = self.noms_text.get("1.0", "end-1c")
            save_noms_propres(noms_content)

            do_restart = False
            if needs_restart:
                do_restart = messagebox.askyesno(
                    "Redémarrage nécessaire",
                    "Tu as changé le modèle, le device ou la précision.\n\n"
                    "Redémarrer maintenant pour appliquer ?"
                )
            else:
                messagebox.showinfo("Sauvegardé", "Paramètres sauvegardés !")

        except Exception as e:
            do_restart = False
            print(f"[config_ui] ERREUR sauvegarde : {e}", flush=True)
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde :\n{e}")

        self.root.destroy()
        if self.on_close_callback:
            self.on_close_callback(needs_restart=do_restart)

    def _save_and_restart(self):
        """Sauvegarde et force le redémarrage de l'application."""
        try:
            lang = self.lang_var.get()
            auto_start = self.autostart_var.get()
            mic_selection = self.mic_var.get()
            mic_value = "" if mic_selection == "(défaut système)" else mic_selection

            new_cfg = {
                "model_size": self.model_var.get(),
                "custom_model_path": self.custom_model_var.get().strip() if self.use_finetuned_var.get() else "",
                "device": self.device_var.get(),
                "compute_type": self.compute_var.get(),
                "language": lang if lang != "auto" else None,
                "audio_gain": round(self.gain_var.get(), 1),
                "auto_paste": self.paste_var.get(),
                "auto_start": auto_start,
                "microphone": mic_value,
                "hotkey_primary": self.hotkey1_var.get(),
                "hotkey_secondary": self.hotkey2_var.get(),
                "stt_engine": self.engine_var.get(),
                "groq_api_key": self.groq_key_var.get().strip(),
                "groq_model": self.groq_model_var.get(),
                "groq_fallback_local": self.groq_fallback_var.get(),
                "moonshine_model": self.moonshine_model_var.get().strip(),
                "moonshine_backend": self.moonshine_backend_var.get(),
                "moonshine_fallback_local": self.moonshine_fallback_var.get(),
                "wav2vec2_model": self.wav2vec2_model_var.get().strip(),
                "wav2vec2_fallback_local": self.wav2vec2_fallback_var.get(),
                "wav2vec2_hotwords": self.wav2vec2_hotwords_var.get(),
                "wav2vec2_hotword_weight": float(self.wav2vec2_hotword_weight_var.get() or 10.0),
                "fastconformer_model": self.fc_model_var.get().strip(),
                "fastconformer_backend": self.fc_backend_var.get(),
                "fastconformer_fallback_local": self.fc_fallback_var.get(),
                "fuzzy_enabled": self.fuzzy_enabled_var.get(),
                "fuzzy_threshold": self.fuzzy_threshold_var.get(),
                "restore_case": self.restore_case_var.get(),
                "restore_case_nouns": self.case_nouns_var.get(),
                "restore_case_sentences": self.case_sentences_var.get(),
                "restore_case_punctuation": self.case_punct_var.get(),
                "restore_case_style": self.case_style_var.get(),
                "api_enabled": self.api_enabled_var.get(),
                "api_host": self.api_host_var.get().strip(),
                "api_port": int(self.api_port_var.get()),
            }
            save_config(new_cfg)

            if auto_start and not _startup_shortcut_exists():
                _create_startup_shortcut()
            elif not auto_start and _startup_shortcut_exists():
                _remove_startup_shortcut()

            vocab_content = self.vocab_text.get("1.0", "end-1c")
            save_vocab(vocab_content)

            corr_content = self.corrections_text.get("1.0", "end-1c")
            save_corrections(corr_content)

            noms_content = self.noms_text.get("1.0", "end-1c")
            save_noms_propres(noms_content)

        except Exception as e:
            print(f"[config_ui] ERREUR sauvegarde : {e}", flush=True)
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde :\n{e}")
            return

        self.root.destroy()
        if self.on_close_callback:
            self.on_close_callback(needs_restart=True)

    def _cancel(self):
        self.root.destroy()
        if self.on_close_callback:
            self.on_close_callback(needs_restart=False)

    def run(self):
        self.root.mainloop()


def open_config_window(on_close_callback=None):
    """Ouvre la fenêtre de config. Peut être appelé depuis un thread."""
    win = ConfigWindow(on_close_callback=on_close_callback)
    win.run()


if __name__ == "__main__":
    # Quand lancé en sous-processus, le code retour indique si un redémarrage est nécessaire
    _exit_code = 0

    def _standalone_callback(needs_restart=False):
        global _exit_code
        _exit_code = 1 if needs_restart else 0

    open_config_window(on_close_callback=_standalone_callback)
    sys.exit(_exit_code)
