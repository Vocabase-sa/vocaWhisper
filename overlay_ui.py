"""
Whisper Dictation - Overlay d'enregistrement
=============================================
Pastille rouge clignotante pendant l'enregistrement.
Utilise une seule instance tkinter persistante (show/hide).
"""

import math
import platform
import threading
import time
import tkinter as tk


IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import ctypes

BG_COLOR = "#1a1a2e"
DOT_COLOR_MAX = (230, 50, 58)
OVERLAY_SIZE = 48
OVERLAY_MARGIN_BOTTOM = 80
FPS = 20
FRAME_MS = 1000 // FPS

_overlay: "RecordingOverlay | None" = None
_lock = threading.Lock()


class RecordingOverlay:
    """Pastille rouge clignotante - instance unique persistante."""

    def __init__(self):
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._visible = False
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3)

    def show(self):
        if self._root and not self._visible:
            self._visible = True
            try:
                self._root.after(0, self._do_show)
            except Exception:
                pass

    def hide(self):
        if self._root and self._visible:
            self._visible = False
            try:
                self._root.after(0, self._do_hide)
            except Exception:
                pass

    def _do_show(self):
        try:
            self._root.deiconify()
        except Exception:
            pass

    def _do_hide(self):
        try:
            self._root.withdraw()
        except Exception:
            pass

    def _run_tk(self):
        self._root = tk.Tk()
        root = self._root

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.92)
        root.configure(bg=BG_COLOR)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - OVERLAY_SIZE) // 2
        y = screen_h - OVERLAY_SIZE - OVERLAY_MARGIN_BOTTOM
        root.geometry(f"{OVERLAY_SIZE}x{OVERLAY_SIZE}+{x}+{y}")

        self._canvas = tk.Canvas(
            root, width=OVERLAY_SIZE, height=OVERLAY_SIZE,
            bg=BG_COLOR, highlightthickness=0, bd=0,
        )
        self._canvas.pack()

        root.update_idletasks()
        self._make_click_through()

        # Commencer caché
        root.withdraw()
        self._ready.set()

        # Boucle d'animation
        root.after(FRAME_MS, self._update_frame)
        root.mainloop()

    def _make_click_through(self):
        """Rend la fenêtre transparente aux clics (Windows uniquement)."""
        if not IS_WINDOWS:
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= 0x00000020 | 0x00080000
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def _update_frame(self):
        if self._visible:
            canvas = self._canvas
            canvas.delete("all")

            pulse = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(time.time() * 4))
            r = 14
            cx, cy = OVERLAY_SIZE // 2, OVERLAY_SIZE // 2
            red = int(DOT_COLOR_MAX[0] * pulse)
            green = int(DOT_COLOR_MAX[1] * pulse)
            blue = int(DOT_COLOR_MAX[2] * pulse)
            color = f"#{red:02x}{green:02x}{blue:02x}"
            canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline="",
            )

        try:
            self._root.after(FRAME_MS, self._update_frame)
        except Exception:
            pass


def _get_overlay() -> RecordingOverlay:
    """Retourne l'instance singleton, la crée si nécessaire."""
    global _overlay
    with _lock:
        if _overlay is None:
            _overlay = RecordingOverlay()
        return _overlay


def show_overlay(audio_levels_deque=None):
    """Affiche l'overlay."""
    overlay = _get_overlay()
    overlay.show()
    return overlay


def hide_overlay(overlay=None):
    """Cache l'overlay."""
    o = _overlay if overlay is None else overlay
    if o:
        o.hide()
