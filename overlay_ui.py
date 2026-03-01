"""
Whisper Dictation - Overlay d'enregistrement
=============================================
Pilule futuriste flottante pendant l'enregistrement.
Affiche un indicateur REC, un timer et des barres audio animees.
Utilise une seule instance tkinter persistante (show/hide).
"""

import collections
import math
import platform
import random
import threading
import time
import tkinter as tk


IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import ctypes

# --- Dimensions ---
OVERLAY_WIDTH = 240
OVERLAY_HEIGHT = 48
OVERLAY_MARGIN_BOTTOM = 80
CORNER_RADIUS = 24

# --- Couleurs ---
BG_COLOR = "#0d0d1a"
GLOW_COLOR_DIM = (80, 20, 25)
GLOW_COLOR_BRIGHT = (230, 50, 58)
REC_DOT_COLOR = "#e6323a"
TEXT_COLOR = "#e0e0e0"
TEXT_REC_COLOR = "#ff4d55"
BAR_COLOR_LOW = (60, 180, 220)
BAR_COLOR_HIGH = (230, 50, 58)

# --- Animation ---
FPS = 25
FRAME_MS = 1000 // FPS
NUM_BARS = 14
BAR_WIDTH = 3
BAR_GAP = 2
BAR_MAX_HEIGHT = 24

_overlay: "RecordingOverlay | None" = None
_lock = threading.Lock()


def _lerp_color(c1: tuple, c2: tuple, t: float) -> str:
    """Interpole lineairement entre deux couleurs RGB."""
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    """Dessine un rectangle aux coins arrondis sur un canvas tkinter."""
    points = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1, x2, y1 + r,
        x2, y2 - r,
        x2, y2, x2 - r, y2,
        x1 + r, y2,
        x1, y2, x1, y2 - r,
        x1, y1 + r,
        x1, y1, x1 + r, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class RecordingOverlay:
    """Pilule futuriste - instance unique persistante."""

    def __init__(self):
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._visible = False
        self._ready = threading.Event()
        self._start_time = 0.0
        self._audio_levels = None
        self._bar_targets = [0.0] * NUM_BARS
        self._bar_current = [0.0] * NUM_BARS
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3)

    def show(self, audio_levels=None):
        self._audio_levels = audio_levels
        self._start_time = time.time()
        self._bar_targets = [0.0] * NUM_BARS
        self._bar_current = [0.0] * NUM_BARS
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
        root.configure(bg="black")

        # Couleur de transparence pour Windows
        if IS_WINDOWS:
            root.attributes("-transparentcolor", "black")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - OVERLAY_WIDTH) // 2
        y = screen_h - OVERLAY_HEIGHT - OVERLAY_MARGIN_BOTTOM
        root.geometry(f"{OVERLAY_WIDTH}x{OVERLAY_HEIGHT}+{x}+{y}")

        self._canvas = tk.Canvas(
            root, width=OVERLAY_WIDTH, height=OVERLAY_HEIGHT,
            bg="black", highlightthickness=0, bd=0,
        )
        self._canvas.pack()

        root.update_idletasks()
        self._make_click_through()

        # Commencer cache
        root.withdraw()
        self._ready.set()

        # Boucle d'animation
        root.after(FRAME_MS, self._update_frame)
        root.mainloop()

    def _make_click_through(self):
        """Rend la fenetre transparente aux clics (Windows uniquement)."""
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
            self._draw()

        try:
            self._root.after(FRAME_MS, self._update_frame)
        except Exception:
            pass

    def _draw(self):
        canvas = self._canvas
        canvas.delete("all")

        t = time.time()
        W, H = OVERLAY_WIDTH, OVERLAY_HEIGHT
        cy = H // 2

        # --- Pulse (0..1) pour les animations ---
        pulse = 0.5 + 0.5 * math.sin(t * 3.5)

        # --- Halo / Glow externe (bordure neon) ---
        glow_color = _lerp_color(GLOW_COLOR_DIM, GLOW_COLOR_BRIGHT, pulse)
        _draw_rounded_rect(canvas, 0, 0, W - 1, H - 1, CORNER_RADIUS,
                           fill="", outline=glow_color, width=2)

        # --- Fond de la capsule ---
        _draw_rounded_rect(canvas, 2, 2, W - 3, H - 3, CORNER_RADIUS - 2,
                           fill=BG_COLOR, outline="")

        # --- Point REC clignotant ---
        rec_dot_visible = math.sin(t * 5) > -0.2
        dot_x = 20
        if rec_dot_visible:
            dot_r = 5
            canvas.create_oval(
                dot_x - dot_r, cy - dot_r, dot_x + dot_r, cy + dot_r,
                fill=REC_DOT_COLOR, outline="",
            )
            # Petit halo autour du point
            halo_r = dot_r + 3
            halo_color = _lerp_color((80, 20, 25), (180, 40, 45), pulse)
            canvas.create_oval(
                dot_x - halo_r, cy - halo_r, dot_x + halo_r, cy + halo_r,
                fill="", outline=halo_color, width=1,
            )

        # --- Texte "REC" ---
        canvas.create_text(
            40, cy, text="REC", anchor="w",
            fill=TEXT_REC_COLOR, font=("Consolas", 10, "bold"),
        )

        # --- Timer ---
        elapsed = t - self._start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        timer_text = f"{minutes}:{seconds:02d}"
        canvas.create_text(
            80, cy, text=timer_text, anchor="w",
            fill=TEXT_COLOR, font=("Consolas", 10),
        )

        # --- Separateur ---
        sep_x = 110
        canvas.create_line(
            sep_x, 10, sep_x, H - 10,
            fill="#333344", width=1,
        )

        # --- Barres audio ---
        self._update_bar_targets()
        bars_start_x = 118
        for i in range(NUM_BARS):
            # Interpolation fluide vers la cible
            self._bar_current[i] += (self._bar_targets[i] - self._bar_current[i]) * 0.3

            bar_h = max(2, int(self._bar_current[i] * BAR_MAX_HEIGHT))
            bx = bars_start_x + i * (BAR_WIDTH + BAR_GAP)
            by_top = cy - bar_h // 2
            by_bot = cy + bar_h // 2

            # Couleur : bleu cyan pour bas, rouge pour haut
            intensity = self._bar_current[i]
            bar_color = _lerp_color(BAR_COLOR_LOW, BAR_COLOR_HIGH, intensity)

            canvas.create_rectangle(
                bx, by_top, bx + BAR_WIDTH, by_bot,
                fill=bar_color, outline="",
            )

    def _update_bar_targets(self):
        """Met a jour les hauteurs cibles des barres depuis les niveaux audio."""
        level = 0.0
        if self._audio_levels and len(self._audio_levels) > 0:
            recent = list(self._audio_levels)[-5:]
            level = sum(recent) / len(recent)
            level = min(level * 3.0, 1.0)

        for i in range(NUM_BARS):
            if level > 0.02:
                # Variation par barre pour un effet "equalizer"
                variation = 0.3 * math.sin(time.time() * 8 + i * 0.7)
                center_factor = 1.0 - abs(i - NUM_BARS / 2) / (NUM_BARS / 2) * 0.4
                target = level * center_factor + variation * level * 0.5
                target = max(0.05, min(1.0, target))
            else:
                # Mode idle : petites ondulations subtiles
                target = 0.05 + 0.04 * math.sin(time.time() * 2 + i * 0.5)
            self._bar_targets[i] = target


def _get_overlay() -> RecordingOverlay:
    """Retourne l'instance singleton, la cree si necessaire."""
    global _overlay
    with _lock:
        if _overlay is None:
            _overlay = RecordingOverlay()
        return _overlay


def show_overlay(audio_levels_deque=None):
    """Affiche l'overlay."""
    overlay = _get_overlay()
    overlay.show(audio_levels=audio_levels_deque)
    return overlay


def hide_overlay(overlay=None):
    """Cache l'overlay."""
    o = _overlay if overlay is None else overlay
    if o:
        o.hide()


if __name__ == "__main__":
    # Test standalone avec simulation audio
    levels = collections.deque(maxlen=50)

    def _simulate_audio():
        while True:
            levels.append(random.random() * 0.5 + 0.1)
            time.sleep(0.05)

    threading.Thread(target=_simulate_audio, daemon=True).start()
    overlay = _get_overlay()
    overlay.show(audio_levels=levels)
    print("Overlay affiche. Ctrl+C pour quitter.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        overlay.hide()
