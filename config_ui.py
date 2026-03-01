"""
Interface de configuration pour Whisper Dictation.
Fenêtre tkinter avec onglets Général et Vocabulaire.
"""

import json
import os
import platform
import subprocess
import sys
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
VOCAB_FILE = os.path.join(BASE_DIR, "vocabulaire.txt")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "corrections.txt")

DEFAULTS = {
    "model_size": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "language": "fr",
    "audio_gain": 10.0,
    "auto_paste": True,
    "auto_start": False,
    "microphone": "",
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
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo",
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
        self.root.title("Whisper Dictation - Paramètres")
        self.root.geometry("520x560")
        self.root.resizable(True, True)
        self.root.minsize(520, 560)

        # Icône de la fenêtre (Vocabase)
        self._set_window_icon()

        # Style
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 4])

        # --- Boutons en bas (packés EN PREMIER avec side=bottom pour toujours être visibles) ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(side="bottom", fill="x")

        save_btn = tk.Button(
            btn_frame,
            text=" Sauvegarder ",
            command=self._save_and_close,
            bg="#28a745",
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
        )
        save_btn.pack(side="right", padx=(5, 0))

        restart_btn = tk.Button(
            btn_frame,
            text=" Redémarrer ",
            command=self._save_and_restart,
            bg="#dc3545",
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
        )
        restart_btn.pack(side="right", padx=(5, 0))

        ttk.Button(btn_frame, text="Annuler", command=self._cancel).pack(side="right")

        # Onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # --- Onglet Général ---
        tab_general = ttk.Frame(notebook, padding=15)
        notebook.add(tab_general, text="Général")

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
            tab_general, text="", font=("Segoe UI", 9), anchor="w", fg="#dc3545",
        )
        self.model_warning_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4), padx=(0, 0))
        self._update_model_status()
        model_combo.bind("<<ComboboxSelected>>", lambda e: self._update_model_status())
        row += 1

        # Device
        ttk.Label(tab_general, text="Device :").grid(row=row, column=0, sticky="w", pady=6)
        self.device_var = tk.StringVar(value=self.cfg["device"])
        device_values = ["cuda", "cpu"]
        if IS_MAC:
            device_values = ["mps", "cpu"]
        device_combo = ttk.Combobox(tab_general, textvariable=self.device_var, state="readonly", width=18,
                                    values=device_values)
        device_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
        ttk.Label(tab_general, text="*redémarrage requis", foreground="gray").grid(row=row, column=2, padx=(5, 0))
        row += 1

        # Compute type
        ttk.Label(tab_general, text="Précision :").grid(row=row, column=0, sticky="w", pady=6)
        self.compute_var = tk.StringVar(value=self.cfg["compute_type"])
        compute_combo = ttk.Combobox(tab_general, textvariable=self.compute_var, state="readonly", width=18,
                                     values=["float16", "float32", "int8"])
        compute_combo.grid(row=row, column=1, sticky="w", pady=6, padx=(10, 0))
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

        # --- Onglet Vocabulaire ---
        tab_vocab = ttk.Frame(notebook, padding=15)
        notebook.add(tab_vocab, text="Vocabulaire")

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

    def _update_model_status(self):
        """Met à jour le label indiquant si le modèle est en local ou à télécharger."""
        model = self.model_var.get()
        if _is_model_cached(model):
            cache_path = _get_model_cache_dir(model)
            size_gb = _get_dir_size_gb(cache_path)
            self.model_status_label.config(
                text=f"  \u2705  En local ({size_gb:.2f} Go)",
                fg="#28a745",
            )
        else:
            expected_gb = MODEL_SIZES_GB.get(model, 0)
            if expected_gb > 0:
                self.model_status_label.config(
                    text=f"  \u2b07  \u00c0 t\u00e9l\u00e9charger (~{expected_gb:.1f} Go)",
                    fg="#dc3545",
                )
            else:
                self.model_status_label.config(
                    text=f"  \u2b07  \u00c0 t\u00e9l\u00e9charger",
                    fg="#dc3545",
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
                    fg="#28a745",
                )
            else:
                self.token_label.config(
                    text=f"Tokens : {token_count} / {max_tokens} utilises   |   {abs(remaining)} en trop (sera tronque)",
                    fg="#dc3545",
                )
        else:
            # Pas de tokenizer disponible : estimation approximative
            lines = text.strip().splitlines()
            words = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
            approx = len(", ".join(words).split()) if words else 0
            self.token_label.config(
                text=f"~{approx} mots (tokenizer non disponible, estimation)",
                fg="#666666",
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
            )

            # Sauvegarder config
            lang = self.lang_var.get()
            auto_start = self.autostart_var.get()
            # Microphone : vide si "(défaut système)"
            mic_selection = self.mic_var.get()
            mic_value = "" if mic_selection == "(défaut système)" else mic_selection

            new_cfg = {
                "model_size": self.model_var.get(),
                "device": self.device_var.get(),
                "compute_type": self.compute_var.get(),
                "language": lang if lang != "auto" else None,
                "audio_gain": round(self.gain_var.get(), 1),
                "auto_paste": self.paste_var.get(),
                "auto_start": auto_start,
                "microphone": mic_value,
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
                "device": self.device_var.get(),
                "compute_type": self.compute_var.get(),
                "language": lang if lang != "auto" else None,
                "audio_gain": round(self.gain_var.get(), 1),
                "auto_paste": self.paste_var.get(),
                "auto_start": auto_start,
                "microphone": mic_value,
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
