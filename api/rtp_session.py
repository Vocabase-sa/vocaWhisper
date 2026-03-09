"""Session RTP individuelle pour un client.

Chaque session accumule l'audio dans un buffer continu.
Pas de modele Whisper permanent — architecture borrow-style :
le modele est emprunte au pool uniquement pendant la transcription.

Inspire de stt-api/routes/rtp_listener.py:RTPSession.
"""

import os
import time
import wave

from api.rtp_config import (
    CHANNELS,
    RECORD_WAV,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SAVE_DIR,
    WEBHOOK_URL,
    logger,
)


class RTPSession:
    """Gestion d'une session RTP individuelle pour un client specifique."""

    def __init__(self, client_addr, client_port, save_dir=None,
                 webhook_url=None, client_id=None, language="fr"):
        # --- Identification ---
        self.client_addr = client_addr
        self.client_port = client_port
        self.client_key = f"{client_addr}:{client_port}"
        self.client_id = client_id or f"client_{client_addr}_{client_port}"
        self.session_id = f"session_{int(time.time())}_{client_addr}_{client_port}"
        self.language = language

        # --- Webhook ---
        self.webhook_url = webhook_url or WEBHOOK_URL

        # --- Buffers ---
        self.audio_buffer = bytearray()        # int16 brut (pre-traitement)
        self.speech_buffer = []                 # list[np.ndarray float32] pour Whisper
        self.speech_buffer_duration = 0.0       # duree cumulee (secondes)

        # --- Etat ---
        self.last_activity = time.time()
        self.current_file_start = time.time()
        self.partial_text = ""
        self.consecutive_timeouts = 0
        self.has_audio_data = False
        self.has_pending_partial = False

        # --- Auto-cloture apres transcription finale ---
        self.final_transcription_sent = False
        self.final_transcription_time = None

        # --- Calibration bruit de fond ---
        self.noise_floor = None
        self.noise_calibration_samples = []
        self.noise_calibrated = False
        self.noise_calibration_start = time.time()
        self.NOISE_CALIBRATION_DURATION = 0.5   # secondes
        self.NOISE_MARGIN = 1.5                 # marge au-dessus du bruit

        # --- Enregistrement WAV (debug) ---
        self.save_dir = save_dir or SAVE_DIR
        self.wav_file = None
        self.wav_path = None
        if RECORD_WAV:
            self._init_wav_file()

        logger.info(
            f"Nouvelle session RTP creee pour {self.client_key} "
            f"(id={self.client_id}, lang={self.language})"
        )

    # ------------------------------------------------------------------
    # WAV recording
    # ------------------------------------------------------------------
    def _init_wav_file(self):
        """Ouvre un nouveau fichier WAV pour l'enregistrement."""
        if self.wav_file:
            self.wav_file.close()
            logger.info(f"Fichier audio ferme pour {self.client_key}: {self.wav_path}")

        timestamp = int(time.time())
        self.current_file_start = timestamp
        client_id_safe = self.client_id.replace(":", "_").replace("/", "_")
        wav_filename = f"audio_{client_id_safe}_{timestamp}.wav"
        self.wav_path = os.path.join(self.save_dir, wav_filename)

        self.wav_file = wave.open(self.wav_path, "wb")
        self.wav_file.setnchannels(CHANNELS)
        self.wav_file.setsampwidth(SAMPLE_WIDTH)
        self.wav_file.setframerate(SAMPLE_RATE)

        logger.info(f"Enregistrement audio demarre pour {self.client_key}: {self.wav_path}")

    def init_new_wav_file(self):
        """Re-initialise le fichier WAV (appele depuis le listener)."""
        self._init_wav_file()

    # ------------------------------------------------------------------
    # Speech buffer
    # ------------------------------------------------------------------
    def clear_speech_buffer(self):
        """Vide le buffer de parole accumule."""
        self.speech_buffer.clear()
        self.speech_buffer_duration = 0.0

    def reset_state(self):
        """Reinitialise l'etat de la session (apres transcription finale)."""
        self.partial_text = ""
        self.has_pending_partial = False
        self.consecutive_timeouts = 0
        self.has_audio_data = False
        self.clear_speech_buffer()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Libere les ressources de la session (pas de modele a liberer)."""
        if self.wav_file:
            try:
                self.wav_file.close()
            except Exception:
                pass
            self.wav_file = None
            logger.info(f"[SESSION-CLEANUP] Fichier audio ferme pour {self.client_key}")

        self.audio_buffer = bytearray()
        self.clear_speech_buffer()
        logger.info(f"[SESSION-CLEANUP] Ressources liberees pour {self.client_key}")
