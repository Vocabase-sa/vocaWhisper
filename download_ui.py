"""
Whisper Dictation - Fenêtre de progression téléchargement/chargement
=====================================================================
Affiche une fenêtre tkinter avec une barre de progression pendant le
téléchargement ou le chargement du modèle.
Fonctionne dans un thread daemon séparé (même pattern que overlay_ui.py).
"""

import os
import platform
import threading
import time
import tkinter as tk
from tkinter import ttk

IS_WINDOWS = platform.system() == "Windows"

# Sur Windows, définir un AppUserModelID pour notre icône dans la barre des tâches
if IS_WINDOWS:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vocabase.vocawhisper.download")

# --- Style ---
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT_COLOR = "#89b4fa"
SUBTEXT_COLOR = "#a6adc8"
PCT_COLOR = "#cdd6f4"
WIN_WIDTH = 420
WIN_HEIGHT = 180

# Temps estimé de chargement GPU par modèle (secondes)
LOAD_TIME_ESTIMATES = {
    "tiny": 2, "tiny.en": 2,
    "base": 3, "base.en": 3,
    "small": 4, "small.en": 4,
    "medium": 6, "medium.en": 6,
    "large-v1": 8, "large-v2": 8, "large-v3": 8,
    "large-v3-turbo": 6,
    "distil-large-v2": 5, "distil-large-v3": 5,
}


class DownloadProgressWindow:
    """Fenêtre de progression pour téléchargement / chargement du modèle."""

    def __init__(self):
        self._root: tk.Tk | None = None
        self._label_title: tk.Label | None = None
        self._label_status: tk.Label | None = None
        self._label_percent: tk.Label | None = None
        self._label_detail: tk.Label | None = None
        self._progress: ttk.Progressbar | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._loading_active = False
        self._loading_start = 0.0
        self._loading_estimate = 8.0
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    @staticmethod
    def _set_window_icon(root):
        """Définit l'icône de la fenêtre depuis icons/icon.ico (priorité .ico sur Windows)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_dir = os.path.join(base_dir, "icons")
        # .ico en priorité (fonctionne pour la barre des tâches Windows)
        for name in ("icon.ico", "icon_green.ico"):
            path = os.path.join(icon_dir, name)
            if os.path.isfile(path):
                try:
                    root.iconbitmap(path)
                    return
                except Exception:
                    pass
        # Fallback : .png via iconphoto
        for name in ("icon.png", "icon_green.png"):
            path = os.path.join(icon_dir, name)
            if os.path.isfile(path):
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(path)
                    photo = ImageTk.PhotoImage(img)
                    root.iconphoto(True, photo)
                    root._icon_ref = photo
                    return
                except Exception:
                    pass

    # --- API publique (thread-safe) ---

    def update_download(self, model_name: str, percent: int, current_gb: float, total_gb: float):
        """Met à jour la progression du téléchargement."""
        if not self._root:
            return
        try:
            self._root.after(0, self._do_update_download, model_name, percent, current_gb, total_gb)
        except Exception:
            pass

    def update_loading(self, model_name: str, device: str):
        """Affiche un message de chargement du modèle (phase après téléchargement)."""
        if not self._root:
            return
        self._loading_active = True
        self._loading_start = time.time()
        self._loading_estimate = LOAD_TIME_ESTIMATES.get(model_name, 8)
        try:
            self._root.after(0, self._do_update_loading, model_name, device)
        except Exception:
            pass

    def update_message(self, title: str, detail: str = ""):
        """Affiche un message générique."""
        if not self._root:
            return
        try:
            self._root.after(0, self._do_update_message, title, detail)
        except Exception:
            pass

    def close(self):
        """Ferme la fenêtre proprement."""
        self._loading_active = False
        if self._root and not self._closed.is_set():
            try:
                self._root.after(0, self._do_close)
            except Exception:
                pass
            self._closed.wait(timeout=3)

    # --- Implémentation interne (tkinter main thread) ---

    def _do_update_download(self, model_name, percent, current_gb, total_gb):
        try:
            self._loading_active = False
            self._label_title.config(text=f"Téléchargement du modèle")
            self._label_status.config(text=f"{model_name}")
            self._progress.stop()
            self._progress.config(mode="determinate")
            self._progress["value"] = percent
            self._label_percent.config(text=f"{percent}%")
            self._label_detail.config(
                text=f"{current_gb:.2f} / {total_gb:.1f} Go"
            )
        except Exception:
            pass

    def _do_update_loading(self, model_name, device):
        try:
            self._label_title.config(text=f"Chargement du modèle")
            self._label_status.config(text=f"{model_name}  →  {device}")
            self._progress.stop()
            self._progress.config(mode="determinate")
            self._progress["value"] = 0
            self._label_percent.config(text="0%")
            self._label_detail.config(text="Initialisation sur le GPU, patience...")
            # Lancer l'animation de progression simulée
            self._tick_loading()
        except Exception:
            pass

    def _tick_loading(self):
        """Met à jour la barre de progression simulée pendant le chargement GPU."""
        if not self._loading_active:
            return
        try:
            elapsed = time.time() - self._loading_start
            # Progression asymptotique : monte vite au début, ralentit vers 95%
            ratio = elapsed / self._loading_estimate
            if ratio < 1.0:
                pct = int(ratio * 85)
            else:
                # Au-delà du temps estimé, monte lentement de 85% à 98%
                extra = (ratio - 1.0) / 3.0  # 3x le temps = 98%
                pct = 85 + int(min(extra, 1.0) * 13)
            pct = min(pct, 98)
            self._progress["value"] = pct
            self._label_percent.config(text=f"{pct}%")
            self._root.after(200, self._tick_loading)
        except Exception:
            pass

    def _do_update_message(self, title, detail):
        try:
            self._loading_active = False
            self._label_title.config(text=title)
            self._label_status.config(text=detail)
            self._progress.stop()
            self._progress.config(mode="indeterminate")
            self._progress.start(15)
            self._label_percent.config(text="")
            self._label_detail.config(text="")
        except Exception:
            pass

    def _do_close(self):
        try:
            self._loading_active = False
            # Flash à 100% avant de fermer
            self._progress.stop()
            self._progress.config(mode="determinate")
            self._progress["value"] = 100
            self._label_percent.config(text="100%")
            self._root.update_idletasks()
            self._root.after(400, self._withdraw_and_quit)
        except Exception:
            self._closed.set()

    def _withdraw_and_quit(self):
        """Cache la fenêtre et quitte mainloop SANS destroy().

        IMPORTANT : root.destroy() détruit l'interpréteur Tcl, ce qui corrompt
        l'état interne de tkinter et empêche toute future utilisation (overlay,
        settings, etc.). On utilise withdraw() + quit() à la place :
        - withdraw() cache la fenêtre immédiatement
        - quit() sort de mainloop() proprement
        - Le thread daemon se termine naturellement → nettoyage automatique
        """
        try:
            self._root.withdraw()
            self._root.quit()
        except Exception:
            pass
        finally:
            self._closed.set()

    # --- Construction de la fenêtre (dans le thread tkinter) ---

    def _run_tk(self):
        self._root = tk.Tk()
        root = self._root
        root.title("Whisper Dictation")
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # Icône de la fenêtre (Vocabase)
        self._set_window_icon(root)

        # Centrer la fenêtre
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - WIN_WIDTH) // 2
        y = (screen_h - WIN_HEIGHT) // 2
        root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{x}+{y}")

        # Empêcher la fermeture par l'utilisateur (sinon l'app crasherait)
        root.protocol("WM_DELETE_WINDOW", lambda: None)

        # --- Style ttk pour la barre de progression ---
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Download.Horizontal.TProgressbar",
            troughcolor="#313244",
            background=ACCENT_COLOR,
            bordercolor=BG_COLOR,
            lightcolor=ACCENT_COLOR,
            darkcolor=ACCENT_COLOR,
            thickness=20,
        )

        # --- Widgets ---
        pad_x = 24
        pad_top = 16

        # Titre (ex: "Téléchargement du modèle")
        self._label_title = tk.Label(
            root, text="Préparation...", font=("Segoe UI", 13, "bold"),
            bg=BG_COLOR, fg=FG_COLOR, anchor="w",
        )
        self._label_title.pack(fill="x", padx=pad_x, pady=(pad_top, 2))

        # Sous-titre (nom du modèle)
        self._label_status = tk.Label(
            root, text="", font=("Segoe UI", 10),
            bg=BG_COLOR, fg=SUBTEXT_COLOR, anchor="w",
        )
        self._label_status.pack(fill="x", padx=pad_x, pady=(0, 8))

        # Frame pour barre + pourcentage sur la même ligne
        bar_frame = tk.Frame(root, bg=BG_COLOR)
        bar_frame.pack(fill="x", padx=pad_x, pady=(0, 6))

        # Barre de progression
        self._progress = ttk.Progressbar(
            bar_frame, length=WIN_WIDTH - 2 * pad_x - 60,
            style="Download.Horizontal.TProgressbar",
            maximum=100, value=0,
        )
        self._progress.pack(side="left", fill="x", expand=True)

        # Pourcentage (affiché à droite de la barre)
        self._label_percent = tk.Label(
            bar_frame, text="0%", font=("Segoe UI", 12, "bold"),
            bg=BG_COLOR, fg=PCT_COLOR, anchor="e", width=5,
        )
        self._label_percent.pack(side="right", padx=(10, 0))

        # Détail (taille téléchargée / message)
        self._label_detail = tk.Label(
            root, text="", font=("Segoe UI", 9),
            bg=BG_COLOR, fg=SUBTEXT_COLOR, anchor="w",
        )
        self._label_detail.pack(fill="x", padx=pad_x, pady=(0, pad_top))

        root.update_idletasks()
        self._ready.set()
        root.mainloop()
