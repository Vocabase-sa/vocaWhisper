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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
VOCAB_FILE = os.path.join(BASE_DIR, "vocabulaire.txt")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "corrections.txt")

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
        self.root.geometry("600x650")
        self.root.resizable(True, True)
        self.root.minsize(600, 600)

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
            tab_general, text="", font=("Segoe UI", 8), anchor="w", fg="#666666",
        )
        self.custom_model_status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4), padx=(20, 0))
        self.custom_model_var.trace_add("write", lambda *_: self._update_custom_model_status())

        # Références aux widgets modèle standard pour les griser
        self.model_combo = model_combo
        # Appliquer l'état initial
        self._toggle_finetuned()
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
            bg="#0d6efd", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
        ).pack(side="right")

        # --- Section 2 : Entraînement ---
        sec_train = ttk.LabelFrame(parent, text="  2. Entraînement  ", padding=8)
        sec_train.pack(fill="x", pady=(0, 6))

        # Ligne 1 : Modèle de base
        row_base = ttk.Frame(sec_train)
        row_base.pack(fill="x", pady=2)
        ttk.Label(row_base, text="Modèle de base :", width=14).pack(side="left")
        self.train_base_model_var = tk.StringVar(value="openai/whisper-large-v3")
        base_combo = ttk.Combobox(row_base, textvariable=self.train_base_model_var, width=38, values=[
            "openai/whisper-large-v3",
            "bofenghuang/whisper-large-v3-french",
            "openai/whisper-large-v2",
            "openai/whisper-medium",
        ])
        base_combo.pack(side="left", padx=(4, 0))

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
            bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
        ).pack(side="right")

        # --- Section 3 : Conversion CTranslate2 ---
        sec_convert = ttk.LabelFrame(parent, text="  3. Conversion CTranslate2  ", padding=8)
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
            bg="#6f42c1", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
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

        # Bouton Arrêter (caché par défaut)
        self._stop_frame = ttk.Frame(log_frame)
        self._stop_frame.pack(fill="x", pady=(4, 0))
        self._stop_btn = tk.Button(
            self._stop_frame, text="Arrêter le processus", command=self._stop_training_process,
            bg="#dc3545", fg="white", font=("Segoe UI", 9),
            relief="flat", padx=8, pady=2, cursor="hand2",
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

        script = os.path.join(BASE_DIR, "fine_tuning", "train.py")
        cmd = [
            python, script,
            "--base_model", self.train_base_model_var.get(),
            "--epochs", self.train_epochs_var.get(),
            "--batch_size", self.train_batch_var.get(),
            "--learning_rate", self.train_lr_var.get(),
        ]
        self._run_subprocess(cmd, "Fine-tuning Whisper")

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
            self.custom_model_status.config(text="", fg="#666666")

    def _update_custom_model_status(self):
        """Met à jour l'indicateur du modèle personnalisé."""
        if not self.use_finetuned_var.get():
            self.custom_model_status.config(text="", fg="#666666")
            return
        path = self.custom_model_var.get().strip()
        if not path:
            self.custom_model_status.config(
                text="  Sélectionnez le dossier du modèle CTranslate2",
                fg="#666666",
            )
        elif os.path.isdir(path):
            # Vérifier si c'est un modèle CTranslate2 valide
            model_bin = os.path.join(path, "model.bin")
            if os.path.isfile(model_bin):
                size_gb = os.path.getsize(model_bin) / 1e9
                self.custom_model_status.config(
                    text=f"  \u2705  Modèle CTranslate2 trouvé ({size_gb:.2f} Go) — *redémarrage requis",
                    fg="#28a745",
                )
            else:
                self.custom_model_status.config(
                    text="  \u26a0  Dossier trouvé mais pas de model.bin",
                    fg="#ff8c00",
                )
        else:
            self.custom_model_status.config(
                text=f"  \u274c  Dossier introuvable : {path}",
                fg="#dc3545",
            )

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
                or (self.custom_model_var.get().strip() if self.use_finetuned_var.get() else "") != self.cfg.get("custom_model_path", "")
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
                "custom_model_path": self.custom_model_var.get().strip() if self.use_finetuned_var.get() else "",
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
