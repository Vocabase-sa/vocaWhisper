"""Pool de modeles Whisper pre-charges pour le streaming RTP.

Architecture borrow-style : les sessions empruntent un modele uniquement
pendant la transcription (~0.3s avec CUDA) puis le rendent immediatement.
Cela permet a 2-3 modeles GPU de servir 15+ sessions paralleles.

Inspire de stt-api/routes/vosk_model_manager.py:ModelPool.
"""

import os
import time
import threading
import numpy as np
from queue import Queue, Empty

from api.rtp_config import (
    DEFAULT_POOL_SIZE,
    SAMPLE_RATE,
    SAVE_DIR,
    logger,
)


class WhisperModelPool:
    """Pool de modeles faster-whisper pre-charges.

    Chaque modele occupe ~3 GB VRAM. Le pool est initialise au demarrage
    et les modeles sont prechauffes pour eliminer la latence du premier appel.
    """

    def __init__(self, size=None):
        self.size = size or DEFAULT_POOL_SIZE
        self.models = []                    # liste de toutes les instances
        self.available_models = Queue()     # file d'attente des modeles disponibles
        self.models_in_use = {}             # {caller_id: (index, model)}
        self.lock = threading.Lock()
        self.initialized = False
        self._model_name = None
        self._device = None
        self._compute_type = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize(self, model_name, device="cuda", compute_type="float16",
                   custom_model_path=""):
        """Pre-charge *size* instances de WhisperModel et prechauffe chacune."""
        if self.initialized:
            logger.info("[POOL] Pool deja initialise")
            return True

        from faster_whisper import WhisperModel

        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type

        # Determiner le chemin effectif du modele
        model_path = model_name
        if custom_model_path and os.path.isdir(custom_model_path):
            model_path = custom_model_path
            logger.info(f"[POOL] Utilisation du modele custom : {custom_model_path}")

        logger.info(
            f"[POOL] Initialisation du pool avec {self.size} modele(s) "
            f"({model_path}, {device}, {compute_type})..."
        )
        start_time = time.time()

        for i in range(self.size):
            try:
                logger.info(f"[POOL] Chargement du modele {i + 1}/{self.size}...")
                model = WhisperModel(model_path, device=device,
                                     compute_type=compute_type)

                # Prechauffer
                self._preheat_model(model, f"pool_{i}")

                self.models.append(model)
                self.available_models.put((i, model))
                logger.info(f"[POOL] Modele {i + 1} charge et pret")

            except Exception as e:
                logger.error(
                    f"[POOL] Erreur lors du chargement du modele {i + 1}: {e}"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"[POOL] Pool initialise en {elapsed:.2f}s — "
            f"{len(self.models)} modele(s) disponible(s)"
        )
        self.initialized = True
        return len(self.models) > 0

    # ------------------------------------------------------------------
    # Prechauffage
    # ------------------------------------------------------------------
    def _preheat_model(self, model, model_id):
        """Prechauffe un modele en effectuant une transcription de test.

        Tente d'abord avec ``recordings/preheating.wav`` (fichier reel),
        puis fall-back sur du bruit synthetique.
        """
        import wave

        try:
            preheating_path = os.path.join(SAVE_DIR, "preheating.wav")

            if os.path.exists(preheating_path):
                try:
                    logger.info(
                        f"[POOL] Prechauffage de {model_id} avec fichier reel : "
                        f"{preheating_path}"
                    )
                    with wave.open(preheating_path, "rb") as wf:
                        if (wf.getnchannels() == 1
                                and wf.getsampwidth() == 2
                                and wf.getframerate() == SAMPLE_RATE):
                            raw = wf.readframes(wf.getnframes())
                            audio_int16 = np.frombuffer(raw, dtype=np.int16)
                            audio_f32 = audio_int16.astype(np.float32) / 32768.0

                            segments, info = model.transcribe(
                                audio_f32,
                                language="fr",
                                beam_size=1,
                                vad_filter=False,
                            )
                            text_parts = [s.text.strip() for s in segments]
                            text = " ".join(text_parts).strip()
                            if text:
                                logger.info(
                                    f"[POOL-PREHEAT] {model_id} — "
                                    f"Reconnaissance : '{text}'"
                                )
                            else:
                                logger.warning(
                                    f"[POOL-PREHEAT] {model_id} — "
                                    f"Aucun texte reconnu dans le fichier de prechauffage"
                                )
                            return True
                        else:
                            logger.warning(
                                f"[POOL] Format audio incompatible pour {model_id}"
                            )
                except Exception as e:
                    logger.warning(
                        f"[POOL] Echec prechauffage avec fichier pour {model_id}: {e}"
                    )

            # Fallback : bruit synthetique (~1s)
            logger.debug(f"[POOL] Prechauffage de {model_id} avec bruit synthetique")
            dummy_audio = np.random.normal(0, 0.01, SAMPLE_RATE).astype(np.float32)
            segments, _ = model.transcribe(
                dummy_audio,
                language="fr",
                beam_size=1,
                vad_filter=False,
            )
            # Consommer le generateur pour forcer l'execution
            _ = list(segments)
            logger.debug(f"[POOL] Modele {model_id} prechauffe")
            return True

        except Exception as e:
            logger.error(f"[POOL] Erreur prechauffage {model_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Acquisition / Liberation
    # ------------------------------------------------------------------
    def acquire_model(self, caller_id, timeout=5.0):
        """Emprunte un modele du pool (bloquant avec timeout).

        *caller_id* est un identifiant libre (ex. ``client_key`` ou thread-id).
        """
        try:
            model_index, model = self.available_models.get(timeout=timeout)

            with self.lock:
                self.models_in_use[caller_id] = (model_index, model)

            logger.debug(
                f"[POOL] Modele #{model_index} emprunte par {caller_id} "
                f"(reste {self.available_models.qsize()} disponible(s))"
            )
            return model_index, model

        except Empty:
            logger.error(
                f"[POOL] Aucun modele disponible pour {caller_id} "
                f"(timeout={timeout}s)"
            )
            return None, None

    def release_model(self, caller_id):
        """Rend un modele au pool."""
        with self.lock:
            if caller_id in self.models_in_use:
                model_index, model = self.models_in_use.pop(caller_id)
                self.available_models.put((model_index, model))
                logger.debug(
                    f"[POOL] Modele #{model_index} rendu par {caller_id} "
                    f"(maintenant {self.available_models.qsize()} disponible(s))"
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    def get_stats(self):
        """Retourne les statistiques du pool."""
        with self.lock:
            return {
                "total": len(self.models),
                "configured_size": self.size,
                "available": self.available_models.qsize(),
                "in_use": len(self.models_in_use),
                "callers": list(self.models_in_use.keys()),
            }


# =====================================================================
# Instance globale
# =====================================================================
whisper_pool = WhisperModelPool()


def initialize_pool(config: dict):
    """Point d'entree pour initialiser le pool depuis server.py."""
    pool_size = config.get("rtp_pool_size", DEFAULT_POOL_SIZE)
    whisper_pool.size = pool_size

    model_name = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    compute_type = config.get("compute_type", "float16")
    custom_path = config.get("custom_model_path", "")

    return whisper_pool.initialize(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        custom_model_path=custom_path,
    )


def get_pool_stats():
    """Raccourci vers les stats du pool global."""
    return whisper_pool.get_stats()
