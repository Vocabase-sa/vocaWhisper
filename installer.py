#!/usr/bin/env python3
"""
VocaWhisper Installer — Assistant d'installation graphique
Interface wizard pour installer VocaWhisper selon le mode choisi.
"""

import json
import os
import platform
import subprocess
import sys
import threading
import shutil
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

PLATFORM_LABEL = "Windows" if IS_WINDOWS else ("macOS" if IS_MAC else "Linux")

# ---------------------------------------------------------------------------
# Paths (defaults — updated at runtime if user changes install dir)
# ---------------------------------------------------------------------------
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SOURCE_DIR  # Will be updated to install_dir


def _venv_bin(venv_dir: str) -> str:
    """Retourne le dossier bin/Scripts du venv selon l'OS."""
    return os.path.join(venv_dir, "Scripts" if IS_WINDOWS else "bin")


def _venv_paths(base: str) -> dict:
    """Calcule tous les chemins venv à partir du dossier d'installation."""
    venv = os.path.join(base, "venv")
    bindir = _venv_bin(venv)
    ext = ".exe" if IS_WINDOWS else ""
    return {
        "venv": venv,
        "python": os.path.join(bindir, f"python{ext}"),
        "pythonw": os.path.join(bindir, f"pythonw{ext}") if IS_WINDOWS else os.path.join(bindir, "python3"),
        "pip": os.path.join(bindir, f"pip{ext}"),
        "config": os.path.join(base, "config.json"),
    }


_paths = _venv_paths(BASE_DIR)
VENV_DIR = _paths["venv"]
VENV_PYTHON = _paths["python"]
VENV_PYTHONW = _paths["pythonw"]
VENV_PIP = _paths["pip"]
CONFIG_PATH = _paths["config"]


def _update_paths(install_dir: str):
    """Met à jour tous les chemins globaux selon le dossier d'installation."""
    global BASE_DIR, VENV_DIR, VENV_PYTHON, VENV_PYTHONW, VENV_PIP, CONFIG_PATH
    BASE_DIR = install_dir
    p = _venv_paths(install_dir)
    VENV_DIR = p["venv"]
    VENV_PYTHON = p["python"]
    VENV_PYTHONW = p["pythonw"]
    VENV_PIP = p["pip"]
    CONFIG_PATH = p["config"]

def _read_version():
    vf = os.path.join(SOURCE_DIR, "VERSION")
    if os.path.exists(vf):
        with open(vf, "r") as f:
            return f.read().strip()
    return "1.5.0"

VERSION = _read_version()


def _find_compatible_python() -> str:
    """Trouve un exécutable Python compatible (3.10-3.12) pour créer le venv.

    PyTorch ne supporte pas encore Python 3.13+, donc on cherche
    spécifiquement une version 3.10, 3.11 ou 3.12.
    """
    # 1) Vérifier si le Python courant est compatible
    vi = sys.version_info
    if 10 <= vi.minor <= 12 and vi.major == 3:
        return sys.executable

    # 2) Sur Windows, essayer le launcher 'py' avec version spécifique
    if IS_WINDOWS:
        for minor in (12, 11, 10):
            try:
                result = subprocess.run(
                    ["py", f"-3.{minor}", "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    return f"py -3.{minor}"
            except Exception:
                continue

    # 3) Essayer python3.12, python3.11, python3.10
    for minor in (12, 11, 10):
        exe = f"python3.{minor}"
        try:
            result = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return exe
        except Exception:
            continue

    # 4) Fallback sur le Python courant (peut échouer avec PyTorch)
    return sys.executable


COMPATIBLE_PYTHON = _find_compatible_python()


# ---------------------------------------------------------------------------
# Theme / Colors
# ---------------------------------------------------------------------------
# Thème Vocabase (vocabase.be)
BG_HEADER = "#0c2d5c"          # Bleu foncé Vocabase
BG_CONTENT = "#f8f8f8"         # Gris très clair
BG_CARD = "#ffffff"            # Blanc
BG_CARD_HOVER = "#fff5f0"     # Corail très pâle
BG_CARD_SELECTED = "#ffe8dd"  # Corail clair sélection
ACCENT = "#F07654"             # Corail/Saumon Vocabase
ACCENT_HOVER = "#d9613f"      # Corail plus foncé au hover
ACCENT_LIGHT = "#ffe8dd"      # Corail pâle
TEXT_DARK = "#0c2d5c"          # Bleu foncé pour titres
TEXT_MID = "#626263"           # Gris moyen
TEXT_LIGHT = "#818181"         # Gris clair
BORDER = "#e2e2e2"            # Bordure grise
BORDER_SELECTED = "#F07654"   # Corail pour sélection
SUCCESS = "#10b981"            # Vert succès
ERROR = "#ef4444"              # Rouge erreur
WHITE = "#ffffff"

if IS_WINDOWS:
    FONT_FAMILY = "Segoe UI"
elif IS_MAC:
    FONT_FAMILY = "SF Pro Text"
else:
    FONT_FAMILY = "Ubuntu"

# ---------------------------------------------------------------------------
# Config templates
# ---------------------------------------------------------------------------
CONFIG_BASE = {
    "custom_model_path": "",
    "language": "fr",
    "audio_gain": 10.0,
    "auto_paste": True,
    "auto_start": False,
    "microphone": "",
    "hotkey_primary": "Ctrl+Space",
    "hotkey_secondary": "Ctrl+F2",
    "groq_api_key": "",
    "groq_model": "whisper-large-v3-turbo",
    "groq_fallback_local": False,
    "fuzzy_enabled": True,
    "fuzzy_threshold": 60,
    "api_enabled": True,
    "api_host": "0.0.0.0",
    "api_port": 5892,
}

MODE_CONFIGS = {
    "groq": {
        **CONFIG_BASE,
        "model_size": "",
        "device": "cpu",
        "compute_type": "float32",
        "stt_engine": "groq",
        "install_mode": "groq",
    },
    "cuda": {
        **CONFIG_BASE,
        "model_size": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "stt_engine": "local",
        "install_mode": "full",
    },
    "mps": {
        **CONFIG_BASE,
        "model_size": "large-v3-turbo",
        "device": "mps",
        "compute_type": "float16",
        "stt_engine": "local",
        "install_mode": "full",
    },
    "cpu": {
        **CONFIG_BASE,
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
        "stt_engine": "local",
        "install_mode": "full",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Installer Application
# ═══════════════════════════════════════════════════════════════════════════
class InstallerApp:
    """Wizard-style installer for VocaWhisper."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VocaWhisper — Installation")
        self.root.resizable(False, False)

        # Window size and centering
        w, h = 750, 650
        sx = (self.root.winfo_screenwidth() - w) // 2
        sy = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{sx}+{sy}")
        self.root.configure(bg=BG_CONTENT)

        # Set window icon
        try:
            if IS_WINDOWS:
                ico_path = os.path.join(SOURCE_DIR, "icons", "icon.ico")
                if os.path.exists(ico_path):
                    self.root.iconbitmap(ico_path)
            else:
                # macOS/Linux : utiliser une image PNG via PhotoImage
                png_path = os.path.join(SOURCE_DIR, "icons", "icon.png")
                if os.path.exists(png_path):
                    self._window_icon = tk.PhotoImage(file=png_path)
                    self.root.iconphoto(True, self._window_icon)
        except Exception:
            pass

        # State
        self.selected_mode = tk.StringVar(value="")
        self.current_step = 0
        self.install_thread = None
        self.install_cancelled = False
        self.install_process = None

        # Build UI skeleton
        self._build_header()
        self._build_content_area()
        self._build_footer()

        # Show first step
        self._show_step(0)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout skeleton
    # ------------------------------------------------------------------
    def _build_header(self):
        """Dark header bar with step indicator."""
        self.header = tk.Frame(self.root, bg=BG_HEADER, height=68)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)

        inner = tk.Frame(self.header, bg=BG_HEADER)
        inner.pack(fill="x", padx=30, pady=0, expand=True)

        self.header_title = tk.Label(
            inner,
            text="VocaWhisper",
            font=(FONT_FAMILY, 16, "bold"),
            fg=WHITE,
            bg=BG_HEADER,
        )
        self.header_title.pack(side="left", pady=18)

        # Step indicator (right side)
        self.step_frame = tk.Frame(inner, bg=BG_HEADER)
        self.step_frame.pack(side="right", pady=18)

        self.step_labels = []
        step_names = ["Accueil", "Mode", "Installation", "Terminé"]
        for i, name in enumerate(step_names):
            lbl = tk.Label(
                self.step_frame,
                text=name,
                font=(FONT_FAMILY, 9),
                fg=TEXT_LIGHT,
                bg=BG_HEADER,
                padx=8,
            )
            lbl.pack(side="left")
            self.step_labels.append(lbl)
            if i < len(step_names) - 1:
                sep = tk.Label(
                    self.step_frame,
                    text="\u203a",
                    font=(FONT_FAMILY, 10),
                    fg=TEXT_LIGHT,
                    bg=BG_HEADER,
                )
                sep.pack(side="left")

    def _build_content_area(self):
        """Central content area where step frames are shown."""
        self.content = tk.Frame(self.root, bg=BG_CONTENT)
        self.content.pack(fill="both", expand=True)

        # Pre-build all step frames (hidden)
        self.steps = []
        for builder in [
            self._build_step_welcome,
            self._build_step_mode,
            self._build_step_install,
            self._build_step_done,
        ]:
            frame = tk.Frame(self.content, bg=BG_CONTENT)
            builder(frame)
            self.steps.append(frame)

    def _build_footer(self):
        """Bottom bar with navigation buttons."""
        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill="x")

        self.footer = tk.Frame(self.root, bg=WHITE, height=56)
        self.footer.pack(fill="x")
        self.footer.pack_propagate(False)

        inner = tk.Frame(self.footer, bg=WHITE)
        inner.pack(fill="x", padx=24, expand=True)

        self.btn_prev = self._make_button(
            inner, "  \u2190  Précédent  ", self._prev_step, style="secondary"
        )
        self.btn_prev.pack(side="left", pady=12)

        self.btn_next = self._make_button(
            inner, "  Suivant  \u2192  ", self._next_step, style="primary"
        )
        self.btn_next.pack(side="right", pady=12)

        self.btn_cancel = self._make_button(
            inner, "  Annuler  ", self._cancel_install, style="danger"
        )
        self.btn_launch = self._make_button(
            inner, "  \u25b6  Lancer VocaWhisper  ", self._launch_app, style="primary"
        )
        self.btn_close = self._make_button(
            inner, "  Fermer  ", self.root.destroy, style="secondary"
        )

    # ------------------------------------------------------------------
    # Button factory
    # ------------------------------------------------------------------
    def _make_button(self, parent, text, command, style="primary"):
        colors = {
            "primary": (ACCENT, WHITE, ACCENT_HOVER),
            "secondary": ("#32373c", WHITE, "#0c2d5c"),
            "danger": (ERROR, WHITE, "#dc2626"),
        }
        bg, fg, hover_bg = colors[style]
        btn = tk.Label(
            parent,
            text=text,
            font=(FONT_FAMILY, 10, "bold" if style == "primary" else "normal"),
            fg=fg,
            bg=bg,
            cursor="hand2",
            padx=16,
            pady=6,
        )
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e, b=btn, c=hover_bg: b.configure(bg=c))
        btn.bind("<Leave>", lambda e, b=btn, c=bg: b.configure(bg=c))
        return btn

    # ------------------------------------------------------------------
    # Step 0 — Welcome
    # ------------------------------------------------------------------
    def _build_step_welcome(self, frame):
        spacer = tk.Frame(frame, bg=BG_CONTENT, height=30)
        spacer.pack()

        # Logo area
        logo_frame = tk.Frame(frame, bg=BG_CONTENT)
        logo_frame.pack(pady=(10, 0))

        # Tenter de charger le logo Vocabase
        logo_path = os.path.join(SOURCE_DIR, "icons", "logo_vocabase.png")
        self._logo_image = None  # Keep reference to prevent GC
        if os.path.exists(logo_path):
            try:
                self._logo_image = tk.PhotoImage(file=logo_path)
                # Redimensionner si trop grand (cible ~96px)
                img_w = self._logo_image.width()
                if img_w > 128:
                    factor = img_w // 96
                    self._logo_image = self._logo_image.subsample(max(factor, 1))
                tk.Label(
                    logo_frame,
                    image=self._logo_image,
                    bg=BG_CONTENT,
                ).pack()
            except Exception:
                self._logo_image = None

        if self._logo_image is None:
            tk.Label(
                logo_frame,
                text="\U0001f399",
                font=(FONT_FAMILY, 48),
                bg=BG_CONTENT,
            ).pack()

        tk.Label(
            logo_frame,
            text="VocaWhisper",
            font=(FONT_FAMILY, 28, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
        ).pack(pady=(4, 0))

        tk.Label(
            logo_frame,
            text="Assistant de dictée vocale intelligent",
            font=(FONT_FAMILY, 13),
            fg=TEXT_MID,
            bg=BG_CONTENT,
        ).pack(pady=(2, 0))

        # Divider
        tk.Frame(frame, bg=ACCENT, height=3, width=60).pack(pady=20)

        # Description
        desc = (
            "Bienvenue dans l'assistant d'installation de VocaWhisper.\n"
            "Ce programme va configurer votre environnement selon le mode\n"
            "de reconnaissance vocale que vous choisirez."
        )
        tk.Label(
            frame,
            text=desc,
            font=(FONT_FAMILY, 11),
            fg=TEXT_MID,
            bg=BG_CONTENT,
            justify="center",
        ).pack(pady=(0, 10))

        # --- Dossier d'installation ---
        install_section = tk.Frame(frame, bg=BG_CONTENT)
        install_section.pack(pady=(10, 0), fill="x", padx=40)

        tk.Label(
            install_section,
            text="Dossier d'installation :",
            font=(FONT_FAMILY, 10, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
            anchor="w",
        ).pack(fill="x")

        path_frame = tk.Frame(install_section, bg=BG_CONTENT)
        path_frame.pack(fill="x", pady=(4, 0))

        self.install_dir_var = tk.StringVar(value=SOURCE_DIR)
        self.install_dir_entry = tk.Entry(
            path_frame,
            textvariable=self.install_dir_var,
            font=(FONT_FAMILY, 10),
            bg="white",
            relief="solid",
            bd=1,
        )
        self.install_dir_entry.pack(side="left", fill="x", expand=True, ipady=4)

        browse_btn = tk.Button(
            path_frame,
            text="Parcourir...",
            font=(FONT_FAMILY, 9),
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._browse_install_dir,
        )
        browse_btn.pack(side="left", padx=(8, 0))

        self.install_dir_hint = tk.Label(
            install_section,
            text="Les fichiers seront installés dans ce dossier",
            font=(FONT_FAMILY, 8),
            fg=TEXT_LIGHT,
            bg=BG_CONTENT,
            anchor="w",
        )
        self.install_dir_hint.pack(fill="x", pady=(2, 0))

        # Info box
        info_frame = tk.Frame(frame, bg="#f0f9ff", highlightbackground="#bae6fd",
                              highlightthickness=1, padx=20, pady=12)
        info_frame.pack(pady=(15, 0))

        # Info Python utilisé pour le venv
        tk.Label(
            info_frame,
            text=f"Version {VERSION}   \u2022   {PLATFORM_LABEL}",
            font=(FONT_FAMILY, 9),
            fg="#0369a1",
            bg="#f0f9ff",
        ).pack()

    # ------------------------------------------------------------------
    # Step 1 — Mode Selection
    # ------------------------------------------------------------------
    def _build_step_mode(self, frame):
        tk.Label(
            frame,
            text="Choisissez votre mode d'installation",
            font=(FONT_FAMILY, 15, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
        ).pack(pady=(20, 4))

        tk.Label(
            frame,
            text="Sélectionnez la configuration adaptée à votre matériel.",
            font=(FONT_FAMILY, 10),
            fg=TEXT_MID,
            bg=BG_CONTENT,
        ).pack(pady=(0, 16))

        cards_frame = tk.Frame(frame, bg=BG_CONTENT)
        cards_frame.pack(padx=24, fill="x")

        # Adapter les cartes selon la plateforme
        if IS_MAC:
            gpu_card = {
                "key": "mps",
                "icon": "\U0001f34e",
                "title": "Apple Silicon (MPS)",
                "subtitle": "Performant \u2022 Hors-ligne",
                "desc": (
                    "Modèle large-v3-turbo local.\n"
                    "Accélération Metal (M1-M4).\n"
                    "~1.6 Go VRAM.\n"
                    "Fallback Groq disponible."
                ),
                "badge": "Complet",
                "badge_color": "#6366f1",
            }
        else:
            gpu_card = {
                "key": "cuda",
                "icon": "\U0001f3ae",
                "title": "GPU NVIDIA (CUDA)",
                "subtitle": "Performant \u2022 Hors-ligne",
                "desc": (
                    "Modèle large-v3-turbo local.\n"
                    "Requiert une carte NVIDIA\n"
                    "avec CUDA (6 Go+ VRAM).\n"
                    "Fallback Groq disponible."
                ),
                "badge": "Complet",
                "badge_color": "#6366f1",
            }

        modes = [
            {
                "key": "groq",
                "icon": "\u2601",
                "title": "Groq Cloud",
                "subtitle": "Léger \u2022 Rapide",
                "desc": (
                    "Transcription via l'API Groq.\n"
                    "Aucun GPU requis.\n"
                    "Nécessite une connexion internet\n"
                    "et une clé API Groq."
                ),
                "badge": "Recommandé",
                "badge_color": ACCENT,
            },
            gpu_card,
            {
                "key": "cpu",
                "icon": "\U0001f5a5",
                "title": "CPU uniquement",
                "subtitle": "Universel \u2022 Plus lent",
                "desc": (
                    "Modèle small local sur CPU.\n"
                    "Fonctionne sur tout PC.\n"
                    "Transcription plus lente.\n"
                    "Fallback Groq disponible."
                ),
                "badge": "",
                "badge_color": TEXT_LIGHT,
            },
        ]

        self.card_widgets = {}
        for i, mode in enumerate(modes):
            cards_frame.columnconfigure(i, weight=1, uniform="card")
            card = self._build_mode_card(cards_frame, mode)
            card.grid(row=0, column=i, padx=6, sticky="nsew")

        # --- Champ clé API Groq (visible quand groq ou cuda/cpu sélectionné) ---
        self.groq_api_frame = tk.Frame(frame, bg=BG_CONTENT)
        self.groq_api_frame.pack(fill="x", padx=40, pady=(16, 0))

        self.groq_api_title = tk.Label(
            self.groq_api_frame,
            text="\U0001f511  Clé API Groq :",
            font=(FONT_FAMILY, 10, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
            anchor="w",
        )
        self.groq_api_title.pack(fill="x")

        api_input_frame = tk.Frame(self.groq_api_frame, bg=BG_CONTENT)
        api_input_frame.pack(fill="x", pady=(4, 0))

        self.groq_api_var = tk.StringVar()
        self.groq_api_entry = tk.Entry(
            api_input_frame,
            textvariable=self.groq_api_var,
            font=(FONT_FAMILY, 10),
            bg="white",
            relief="solid",
            bd=1,
            show="*",
        )
        self.groq_api_entry.pack(side="left", fill="x", expand=True, ipady=4)

        self.groq_api_show_var = tk.BooleanVar(value=False)
        show_btn = tk.Checkbutton(
            api_input_frame,
            text="Afficher",
            variable=self.groq_api_show_var,
            font=(FONT_FAMILY, 8),
            bg=BG_CONTENT,
            command=self._toggle_groq_key_visibility,
        )
        show_btn.pack(side="left", padx=(8, 0))

        self.groq_api_hint = tk.Label(
            self.groq_api_frame,
            text="Obtenez votre clé gratuite sur console.groq.com",
            font=(FONT_FAMILY, 8),
            fg=TEXT_LIGHT,
            bg=BG_CONTENT,
            anchor="w",
        )
        self.groq_api_hint.pack(fill="x", pady=(2, 0))

        # Cacher par défaut, affiché quand un mode est sélectionné
        self.groq_api_frame.pack_forget()

    def _build_mode_card(self, parent, mode):
        """Build a selectable card widget for a mode."""
        key = mode["key"]

        outer = tk.Frame(parent, bg=BORDER, padx=2, pady=2)
        card = tk.Frame(outer, bg=BG_CARD, padx=14, pady=14)
        card.pack(fill="both", expand=True)

        # Badge
        if mode["badge"]:
            badge = tk.Label(
                card,
                text=f" {mode['badge']} ",
                font=(FONT_FAMILY, 8, "bold"),
                fg=WHITE,
                bg=mode["badge_color"],
            )
            badge.pack(anchor="e", pady=(0, 4))
        else:
            tk.Frame(card, bg=BG_CARD, height=22).pack()

        # Icon
        tk.Label(
            card,
            text=mode["icon"],
            font=(FONT_FAMILY, 32),
            bg=BG_CARD,
        ).pack(pady=(2, 4))

        # Title
        tk.Label(
            card,
            text=mode["title"],
            font=(FONT_FAMILY, 12, "bold"),
            fg=TEXT_DARK,
            bg=BG_CARD,
        ).pack()

        # Subtitle
        tk.Label(
            card,
            text=mode["subtitle"],
            font=(FONT_FAMILY, 9),
            fg=ACCENT,
            bg=BG_CARD,
        ).pack(pady=(0, 8))

        # Description
        tk.Label(
            card,
            text=mode["desc"],
            font=(FONT_FAMILY, 9),
            fg=TEXT_MID,
            bg=BG_CARD,
            justify="center",
        ).pack()

        # Store references for highlight toggling
        self.card_widgets[key] = {"outer": outer, "card": card, "children": card.winfo_children()}

        # Click binding
        def on_select(event, k=key):
            self.selected_mode.set(k)
            self._update_card_highlights()

        for widget in [outer, card] + card.winfo_children():
            widget.bind("<Button-1>", on_select)
            widget.configure(cursor="hand2")

        # Hover
        def on_enter(event, k=key):
            if self.selected_mode.get() != k:
                outer.configure(bg=ACCENT_LIGHT)

        def on_leave(event, k=key):
            if self.selected_mode.get() != k:
                outer.configure(bg=BORDER)

        for widget in [outer, card] + card.winfo_children():
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        return outer

    def _toggle_groq_key_visibility(self):
        """Bascule l'affichage du champ clé API Groq."""
        if self.groq_api_show_var.get():
            self.groq_api_entry.config(show="")
        else:
            self.groq_api_entry.config(show="*")

    def _update_card_highlights(self):
        sel = self.selected_mode.get()
        for key, widgets in self.card_widgets.items():
            if key == sel:
                widgets["outer"].configure(bg=BORDER_SELECTED)
                widgets["card"].configure(bg=BG_CARD_SELECTED)
                for child in widgets["card"].winfo_children():
                    try:
                        child.configure(bg=BG_CARD_SELECTED)
                    except tk.TclError:
                        pass
            else:
                widgets["outer"].configure(bg=BORDER)
                widgets["card"].configure(bg=BG_CARD)
                for child in widgets["card"].winfo_children():
                    try:
                        child.configure(bg=BG_CARD)
                    except tk.TclError:
                        pass

        # Afficher le champ clé API Groq pour tous les modes
        # (Groq est toujours disponible, en principal ou en fallback)
        if sel:
            self.groq_api_frame.pack(fill="x", padx=40, pady=(16, 0))
            if sel == "groq":
                self.groq_api_title.config(text="\U0001f511  Clé API Groq (requise) :")
            else:
                self.groq_api_title.config(text="\U0001f511  Clé API Groq (optionnelle, pour fallback) :")

    # ------------------------------------------------------------------
    # Step 2 — Installation
    # ------------------------------------------------------------------
    def _build_step_install(self, frame):
        tk.Label(
            frame,
            text="Installation en cours...",
            font=(FONT_FAMILY, 15, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
        ).pack(pady=(18, 4))

        self.install_status = tk.Label(
            frame,
            text="Préparation...",
            font=(FONT_FAMILY, 10),
            fg=TEXT_MID,
            bg=BG_CONTENT,
        )
        self.install_status.pack(pady=(0, 10))

        # Progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Teal.Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background=ACCENT,
            thickness=8,
        )
        self.progress = ttk.Progressbar(
            frame,
            style="Teal.Horizontal.TProgressbar",
            orient="horizontal",
            length=660,
            mode="determinate",
            maximum=100,
        )
        self.progress.pack(pady=(0, 10))

        # Log area
        log_frame = tk.Frame(frame, bg=BORDER, padx=1, pady=1)
        log_frame.pack(padx=30, fill="both", expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg="#1e293b",
            fg="#a5f3fc",
            insertbackground="#a5f3fc",
            relief="flat",
            wrap="word",
            state="disabled",
            height=14,
        )
        self.log_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Step 3 — Done
    # ------------------------------------------------------------------
    def _build_step_done(self, frame):
        spacer = tk.Frame(frame, bg=BG_CONTENT, height=40)
        spacer.pack()

        self.done_icon = tk.Label(
            frame,
            text="\u2714",
            font=(FONT_FAMILY, 56),
            fg=SUCCESS,
            bg=BG_CONTENT,
        )
        self.done_icon.pack()

        self.done_title = tk.Label(
            frame,
            text="Installation terminée avec succès !",
            font=(FONT_FAMILY, 17, "bold"),
            fg=TEXT_DARK,
            bg=BG_CONTENT,
        )
        self.done_title.pack(pady=(8, 4))

        self.done_subtitle = tk.Label(
            frame,
            text="",
            font=(FONT_FAMILY, 10),
            fg=TEXT_MID,
            bg=BG_CONTENT,
        )
        self.done_subtitle.pack(pady=(0, 16))

        # Summary frame
        self.done_summary_frame = tk.Frame(
            frame, bg=WHITE, highlightbackground=BORDER, highlightthickness=1,
            padx=24, pady=16
        )
        self.done_summary_frame.pack(padx=60)

        self.done_summary = tk.Label(
            self.done_summary_frame,
            text="",
            font=(FONT_FAMILY, 10),
            fg=TEXT_MID,
            bg=WHITE,
            justify="left",
        )
        self.done_summary.pack()

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------
    def _show_step(self, idx):
        # Hide all
        for s in self.steps:
            s.pack_forget()

        self.current_step = idx
        self.steps[idx].pack(fill="both", expand=True)

        # Update header step indicator
        for i, lbl in enumerate(self.step_labels):
            if i < idx:
                lbl.configure(fg=ACCENT)
            elif i == idx:
                lbl.configure(fg=WHITE, font=(FONT_FAMILY, 9, "bold"))
            else:
                lbl.configure(fg=TEXT_LIGHT, font=(FONT_FAMILY, 9))

        # Update footer buttons visibility
        self.btn_prev.pack_forget()
        self.btn_next.pack_forget()
        self.btn_cancel.pack_forget()
        self.btn_launch.pack_forget()
        self.btn_close.pack_forget()

        if idx == 0:
            self.btn_next.pack(side="right", pady=12)
        elif idx == 1:
            self.btn_prev.pack(side="left", pady=12)
            self.btn_next.pack(side="right", pady=12)
        elif idx == 2:
            self.btn_cancel.pack(side="right", pady=12)
        elif idx == 3:
            self.btn_launch.pack(side="right", pady=12)
            self.btn_close.pack(side="left", pady=12)

    def _next_step(self):
        if self.current_step == 0:
            # Valider et appliquer le chemin d'installation
            install_dir = self.install_dir_var.get().strip()
            if install_dir:
                _update_paths(os.path.normpath(install_dir))
            self._show_step(1)
        elif self.current_step == 1:
            if not self.selected_mode.get():
                # Flash a message — highlight cards
                return
            self._show_step(2)
            self._start_install()
        elif self.current_step < 3:
            self._show_step(self.current_step + 1)

    def _prev_step(self):
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    # ------------------------------------------------------------------
    # Installation logic
    # ------------------------------------------------------------------
    def _log(self, msg, tag=None):
        """Append message to log widget (thread-safe via after) + fichier log."""
        # Écriture dans le fichier log (toujours, même si le widget plante)
        log_file = os.path.join(SOURCE_DIR, "install.log")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                prefix = {"step": "[STEP]", "error": "[ERR]", "success": "[OK]"}.get(tag, "")
                f.write(f"{prefix} {msg}\n")
        except Exception:
            pass

        def _do():
            self.log_text.configure(state="normal")
            if tag == "step":
                self.log_text.insert("end", f"\n{'='*60}\n", "sep")
                self.log_text.insert("end", f"  {msg}\n", "step")
                self.log_text.insert("end", f"{'='*60}\n", "sep")
            elif tag == "error":
                self.log_text.insert("end", f"[ERREUR] {msg}\n", "err")
            elif tag == "success":
                self.log_text.insert("end", f"[OK] {msg}\n", "ok")
            else:
                self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, _do)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.install_status.configure(text=msg))

    def _set_progress(self, value):
        self.root.after(0, lambda: self.progress.configure(value=value))

    def _run_command(self, cmd, label):
        """Run a subprocess command, streaming output to the log. Returns True on success."""
        self._log(label, tag="step")
        self._set_status(label)

        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=BASE_DIR,
            )
            if IS_WINDOWS:
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self.install_process = subprocess.Popen(cmd, **popen_kwargs)

            for line in self.install_process.stdout:
                if self.install_cancelled:
                    self.install_process.terminate()
                    return False
                stripped = line.rstrip()
                if stripped:
                    self._log(stripped)

            self.install_process.wait()
            rc = self.install_process.returncode
            self.install_process = None

            if rc != 0:
                self._log(f"Commande terminée avec le code {rc}", tag="error")
                return False

            self._log("Terminé.", tag="success")
            return True

        except Exception as exc:
            self._log(str(exc), tag="error")
            return False

    def _browse_install_dir(self):
        """Ouvre un sélecteur de dossier pour choisir le chemin d'installation."""
        path = filedialog.askdirectory(
            title="Choisir le dossier d'installation",
            initialdir=self.install_dir_var.get(),
        )
        if path:
            self.install_dir_var.set(path)

    def _prepare_install_dir(self):
        """Prépare le dossier d'installation : copie les fichiers sources si nécessaire."""
        install_dir = self.install_dir_var.get().strip()
        if not install_dir:
            install_dir = SOURCE_DIR

        # Normaliser le chemin
        install_dir = os.path.normpath(install_dir)
        _update_paths(install_dir)

        # Si c'est le même dossier que les sources, rien à copier
        if os.path.normpath(install_dir) == os.path.normpath(SOURCE_DIR):
            return True

        # Créer le dossier cible si nécessaire
        os.makedirs(install_dir, exist_ok=True)

        # Copier les fichiers du projet (ignorer les dossiers lourds / non nécessaires)
        skip_dirs = {
            "venv", ".venv", "__pycache__", ".git", ".claude",
            "node_modules", "logs", "recordings", "sql",
        }
        # Fichiers essentiels à copier (pas tout le repo)
        essential_extensions = {
            ".py", ".txt", ".json", ".bat", ".vbs", ".sh", ".command",
            ".md", ".png", ".ico", ".example",
        }
        essential_dirs = {"api", "icons", "fine_tuning", "batch", "templates"}

        # Sous-dossiers lourds à ignorer dans la copie
        ignore_subdirs = {"data", "dataset_prepared", "model_ct2", "output",
                          "__pycache__", ".git"}

        for item in os.listdir(SOURCE_DIR):
            if item in skip_dirs:
                continue
            src = os.path.join(SOURCE_DIR, item)
            dst = os.path.join(install_dir, item)
            try:
                if os.path.isdir(src):
                    if item in essential_dirs:
                        # Copier uniquement les scripts, pas les données lourdes
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                            *ignore_subdirs, "*.pyc", "*.bin", "*.safetensors",
                            "*.pt", "*.onnx", "*.wav", "*.mp3"))
                else:
                    # Ne pas écraser les fichiers utilisateur
                    preserve_files = {"config.json", "noms_propres.txt", "vocabulaire.txt", "corrections.txt"}
                    if item in preserve_files and os.path.exists(dst):
                        continue
                    ext = os.path.splitext(item)[1].lower()
                    if ext in essential_extensions:
                        shutil.copy2(src, dst)
            except Exception as exc:
                self._log(f"  Avertissement copie {item}: {exc}")

        self._log(f"Fichiers copiés vers {install_dir}", tag="success")
        return True

    def _deploy_templates(self):
        """Copie les fichiers templates vers le dossier d'installation si absents."""
        templates_dir = os.path.join(SOURCE_DIR, "templates")
        if not os.path.isdir(templates_dir):
            return

        # Fichiers racine : ne copier que si le fichier n'existe pas déjà
        for fname in ("vocabulaire.txt", "corrections.txt", "noms_propres.txt"):
            src = os.path.join(templates_dir, fname)
            dst = os.path.join(BASE_DIR, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                self._log(f"  Template créé : {fname}")

        # fine_tuning/data/ : CSV template + dossier audio vide
        ft_data = os.path.join(BASE_DIR, "fine_tuning", "data")
        os.makedirs(ft_data, exist_ok=True)

        csv_src = os.path.join(templates_dir, "transcriptions.csv")
        csv_dst = os.path.join(ft_data, "transcriptions.csv")
        if os.path.exists(csv_src) and not os.path.exists(csv_dst):
            shutil.copy2(csv_src, csv_dst)
            self._log("  Template créé : fine_tuning/data/transcriptions.csv")

        audio_dir = os.path.join(ft_data, "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir, exist_ok=True)
            self._log("  Dossier créé : fine_tuning/data/audio/")

        # recordings/ : dossier pour les enregistrements WAV
        rec_dir = os.path.join(BASE_DIR, "recordings")
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir, exist_ok=True)
            self._log("  Dossier créé : recordings/")

    def _start_install(self):
        self.install_cancelled = False

        # Configure log tags
        self.log_text.tag_configure("sep", foreground="#475569")
        self.log_text.tag_configure("step", foreground="#67e8f9", font=("Consolas", 9, "bold"))
        self.log_text.tag_configure("err", foreground="#fca5a5")
        self.log_text.tag_configure("ok", foreground="#86efac")

        self.install_thread = threading.Thread(target=self._install_worker, daemon=True)
        self.install_thread.start()

    def _install_worker(self):
        mode = self.selected_mode.get()

        mode_labels = {
            "groq": "Groq Cloud",
            "cuda": "GPU NVIDIA (CUDA)",
            "mps": "Apple Silicon (MPS)",
            "cpu": "CPU",
        }
        self._log(f"Mode sélectionné : {mode_labels.get(mode, mode)}", tag="step")

        # Préparer le dossier d'installation
        self._set_status("Préparation du dossier d'installation...")
        self._log(f"Dossier : {BASE_DIR}", tag="step")
        try:
            self._prepare_install_dir()
        except Exception as exc:
            self._log(f"Erreur préparation dossier : {exc}", tag="error")
            self._install_failed()
            return

        # Determine total steps
        if mode == "groq":
            total_steps = 4  # venv, pip upgrade, requirements-base, config
        else:
            total_steps = 6  # venv, pip upgrade, torch, requirements-base, requirements-local, config
            # mode mps/cuda/cpu all have the same number of steps

        step = 0

        # --- Step: Create venv ---
        step += 1
        self._set_progress((step - 1) / total_steps * 100)

        # Vérifier si le venv existant est compatible (Python 3.10-3.12)
        venv_needs_recreate = False
        if os.path.exists(VENV_PYTHON):
            try:
                result = subprocess.run(
                    [VENV_PYTHON, "-c", "import sys; print(sys.version_info.minor)"],
                    capture_output=True, text=True, timeout=10,
                )
                venv_minor = int(result.stdout.strip())
                if venv_minor > 12 and mode != "groq":
                    self._log(f"Venv existant utilise Python 3.{venv_minor} (incompatible PyTorch).", tag="error")
                    self._log("Recréation du venv avec Python 3.12...", tag="step")
                    venv_needs_recreate = True
                    import shutil as _shutil
                    _shutil.rmtree(VENV_DIR)
            except Exception:
                pass

        if not os.path.exists(VENV_PYTHON) or venv_needs_recreate:
            # Construire la commande venv avec le bon Python
            if COMPATIBLE_PYTHON.startswith("py "):
                # "py -3.12" → split en ["py", "-3.12", "-m", "venv", ...]
                venv_cmd = COMPATIBLE_PYTHON.split() + ["-m", "venv", VENV_DIR]
            else:
                venv_cmd = [COMPATIBLE_PYTHON, "-m", "venv", VENV_DIR]

            self._log(f"Python sélectionné : {COMPATIBLE_PYTHON}")
            ok = self._run_command(venv_cmd, "Création de l'environnement virtuel...")
            if not ok or self.install_cancelled:
                self._install_failed()
                return
        else:
            self._log("Environnement virtuel existant détecté.", tag="success")
            self._set_status("Environnement virtuel OK")

        # --- Step: Upgrade pip ---
        step += 1
        self._set_progress((step - 1) / total_steps * 100)
        ok = self._run_command(
            [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"],
            "Mise à jour de pip..."
        )
        if not ok or self.install_cancelled:
            self._install_failed()
            return

        # --- Step: PyTorch (CUDA / MPS / CPU) ---
        if mode in ("cuda", "mps", "cpu"):
            step += 1
            self._set_progress((step - 1) / total_steps * 100)

            if mode == "cuda":
                torch_cmd = [VENV_PIP, "install", "torch", "torchvision", "torchaudio",
                             "--index-url", "https://download.pytorch.org/whl/cu124"]
                torch_label = "Installation de PyTorch (CUDA 12.4)..."
            elif mode == "mps":
                # macOS : pip install torch installe automatiquement le support MPS
                torch_cmd = [VENV_PIP, "install", "torch", "torchvision", "torchaudio"]
                torch_label = "Installation de PyTorch (MPS / Metal)..."
            else:
                torch_cmd = [VENV_PIP, "install", "torch", "torchvision", "torchaudio",
                             "--index-url", "https://download.pytorch.org/whl/cpu"]
                torch_label = "Installation de PyTorch (CPU)..."

            ok = self._run_command(torch_cmd, torch_label)
            if not ok or self.install_cancelled:
                self._install_failed()
                return

        # --- Step: requirements-base.txt ---
        step += 1
        self._set_progress((step - 1) / total_steps * 100)
        ok = self._run_command(
            [VENV_PIP, "install", "-r", os.path.join(BASE_DIR, "requirements-base.txt")],
            "Installation des dépendances de base..."
        )
        if not ok or self.install_cancelled:
            self._install_failed()
            return

        # --- Step: requirements-local.txt ---
        if mode in ("cuda", "mps", "cpu"):
            step += 1
            self._set_progress((step - 1) / total_steps * 100)
            ok = self._run_command(
                [VENV_PIP, "install", "-r", os.path.join(BASE_DIR, "requirements-local.txt")],
                "Installation des dépendances locales (faster-whisper)..."
            )
            if not ok or self.install_cancelled:
                self._install_failed()
                return

        # --- Step: Write config.json ---
        step += 1
        self._set_progress((step - 1) / total_steps * 100)
        self._log("Écriture de la configuration...", tag="step")
        self._set_status("Écriture de config.json...")

        try:
            new_config = MODE_CONFIGS[mode]
            # Injecter la clé API Groq saisie par l'utilisateur
            groq_key = self.groq_api_var.get().strip()
            if groq_key:
                new_config["groq_api_key"] = groq_key
            # Si un config.json existe déjà, préserver les réglages utilisateur
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                # Ne mettre à jour que les clés liées au mode d'installation
                mode_keys = {"install_mode", "stt_engine", "device", "compute_type",
                              "model_size", "groq_fallback_local"}
                for k in mode_keys:
                    if k in new_config:
                        existing[k] = new_config[k]
                # Ajouter les clés manquantes (nouvelles fonctionnalités)
                for k, v in new_config.items():
                    if k not in existing:
                        existing[k] = v
                config = existing
                self._log("config.json existant mis à jour (réglages préservés)", tag="success")
            else:
                config = new_config
                self._log(f"config.json créé ({mode})", tag="success")
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            self._log(f"Erreur écriture config : {exc}", tag="error")
            self._install_failed()
            return

        # --- Step: Deploy template files ---
        self._set_status("Installation des fichiers modèles...")
        self._deploy_templates()

        # --- Done ---
        self._set_progress(100)
        self._set_status("Installation terminée !")
        self._log("\nInstallation terminée avec succès !", tag="success")

        # Prepare summary for done page
        summary_lines = [
            f"Mode : {mode_labels.get(mode, mode)}",
            f"Moteur STT : {config['stt_engine']}",
        ]
        if mode != "groq":
            summary_lines.append(f"Modèle : {config['model_size']}")
            summary_lines.append(f"Device : {config['device']} ({config['compute_type']})")
        summary_lines.append(f"Configuration : {CONFIG_PATH}")

        summary_text = "\n".join(summary_lines)

        def go_done():
            self.done_subtitle.configure(text="Votre environnement est prêt.")
            self.done_summary.configure(text=summary_text)
            self._show_step(3)

        self.root.after(300, go_done)

    def _install_failed(self):
        if self.install_cancelled:
            self._log("\nInstallation annulée par l'utilisateur.", tag="error")
            self._set_status("Installation annulée")
        else:
            self._log("\nL'installation a échoué.", tag="error")
            self._set_status("Échec de l'installation")

        def show_fail():
            self.done_icon.configure(text="\u2718", fg=ERROR)
            self.done_title.configure(text="Installation échouée" if not self.install_cancelled
                                       else "Installation annulée")
            self.done_subtitle.configure(
                text="Consultez le journal ci-dessus pour plus de détails."
                if not self.install_cancelled
                else "L'installation a été interrompue."
            )
            self.done_summary.configure(text="Vous pouvez relancer l'installateur pour réessayer.")
            self._show_step(3)

        self.root.after(500, show_fail)

    def _cancel_install(self):
        self.install_cancelled = True
        if self.install_process:
            try:
                self.install_process.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Launch app
    # ------------------------------------------------------------------
    def _launch_app(self):
        script = os.path.join(BASE_DIR, "whisper_dictation.py")
        kwargs = {"cwd": BASE_DIR}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen([VENV_PYTHONW, script], **kwargs)
        except Exception:
            subprocess.Popen([VENV_PYTHON, script], **kwargs)
        self.root.destroy()

    # ------------------------------------------------------------------
    # Close handling
    # ------------------------------------------------------------------
    def _on_close(self):
        if self.current_step == 2 and self.install_thread and self.install_thread.is_alive():
            self._cancel_install()
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = InstallerApp()
    app.run()
