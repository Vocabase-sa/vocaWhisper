"""Configuration RTP pour VocaWhisper.

Constantes reprises de stt-api/routes/rtp_config.py, adaptees pour Whisper.
"""

import os
import logging

# --- Repertoire de base du projet ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Audio ---
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2       # 16-bit
CHANNELS = 1            # mono

# --- RTP ---
RTP_PORT = 5002
BUFFER_THRESHOLD = 8192  # ~0.25s a 16kHz 16-bit mono

# --- Detection de silence ---
SILENCE_THRESHOLD_MEAN = 30  # Seuil bas pour etre sensible aux sons faibles

# --- Enregistrement WAV (debug) ---
RECORD_WAV = False
SAVE_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Pool Whisper ---
DEFAULT_POOL_SIZE = 2

# --- Webhook ---
WEBHOOK_URL = ""

# --- Echantillons de debug ---
SAVE_DEBUG_SAMPLES = False
DEBUG_SAMPLES_DIR = os.path.join(SAVE_DIR, "debug_samples")
if SAVE_DEBUG_SAMPLES:
    os.makedirs(DEBUG_SAMPLES_DIR, exist_ok=True)

# --- Logger dedie ---
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rtp_listener")

# Fichier de log dedie
_log_file = os.path.join(LOG_DIR, "rtp_debug.log")
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_file_handler)

logger.info("================ DEMARRAGE DU MODULE RTP WHISPER ================")


def apply_config(config: dict):
    """Applique les valeurs de config.json aux constantes du module.

    Appelee une fois au demarrage par server.py apres chargement de config.json.
    """
    import api.rtp_config as _self

    _self.RTP_PORT = config.get("rtp_port", RTP_PORT)
    _self.RECORD_WAV = config.get("rtp_record_wav", RECORD_WAV)
    _self.WEBHOOK_URL = config.get("rtp_webhook_url", WEBHOOK_URL)
    _self.DEFAULT_POOL_SIZE = config.get("rtp_pool_size", DEFAULT_POOL_SIZE)

    save_dir = config.get("rtp_save_dir", "")
    if save_dir:
        _self.SAVE_DIR = os.path.abspath(save_dir)
        os.makedirs(_self.SAVE_DIR, exist_ok=True)

    logger.info(
        f"[CONFIG] RTP port={_self.RTP_PORT}, pool={_self.DEFAULT_POOL_SIZE}, "
        f"record_wav={_self.RECORD_WAV}, webhook={_self.WEBHOOK_URL or '(none)'}"
    )
