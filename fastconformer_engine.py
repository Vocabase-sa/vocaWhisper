"""
Moteur de transcription FastConformer (NeMo, ONNX ou .nemo).
=============================================================

FastConformer est l'architecture ASR de NVIDIA : un Conformer à sous-échantillonnage
agressif (8x contre 4x), ce qui divise par deux le coût de l'encodeur. Les modèles
français disponibles sont entraînés sur des milliers d'heures, bien plus que les
alternatives légères — d'où une qualité nettement supérieure à taille comparable.

Deux backends :
    - onnx (défaut) : via onnx-asr + onnxruntime. Léger, CPU, aucune dépendance
      lourde. C'est le backend recommandé.
    - nemo : via nemo_toolkit, nécessaire pour les modèles publiés uniquement au
      format .nemo (dont LinTO). Installation lourde (~2-3 Go).

Modèles français connus :
    OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx   (ONNX, défaut)
        Export ONNX du FastConformer FR de NVIDIA. Variante « pc » : restitue
        la ponctuation et les majuscules, contrairement à un CTC nu.
    linagora/linto_stt_fr_fastconformer                     (.nemo, backend nemo)
        115M paramètres, 9500 h de français sur 22+ sources, CC-BY-4.0.
        WER annoncés : 4,70 % (MLS), 8,96 % (CommonVoice), 10,83 % (VoxPopuli).

LIMITE — pas d'initial_prompt :
    Comme Moonshine et wav2vec2, aucun conditionnement possible. vocabulaire.txt
    n'est pas injecté ; corrections.txt et le fuzzy s'appliquent en aval.
"""

import os
import threading
import time

import numpy as np

SAMPLE_RATE = 16000

# Le format d'un modèle dicte son runtime : un .onnx ne se charge pas avec NeMo,
# ni un .nemo avec onnxruntime. Le défaut dépend donc du backend choisi.
DEFAULT_MODELS = {
    "onnx": "OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx",
    "nemo": "linagora/linto_stt_fr_fastconformer",
}
DEFAULT_MODEL = DEFAULT_MODELS["onnx"]

# Type d'architecture attendu par onnx-asr pour les exports NeMo CTC.
ONNX_MODEL_TYPE = "nemo-conformer-ctc"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_CHUNK_SECONDS = 30.0


def _default_log(msg):
    print(msg)


class FastConformerEngine:
    """Encapsule le chargement et l'inférence d'un modèle FastConformer."""

    def __init__(self, model_id=None, backend="onnx", quantization="int8", log=None):
        self.backend = backend if backend in DEFAULT_MODELS else "onnx"
        self.model_id = model_id or DEFAULT_MODELS[self.backend]
        self.quantization = quantization
        self.log = log or _default_log
        self.model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def load(self):
        """Charge le modèle. Idempotent."""
        if self.model is not None:
            return True

        with self._lock:
            if self.model is not None:
                return True

            t0 = time.perf_counter()
            self.log(f"[FastConformer] Chargement de {self.model_id} ({self.backend})...")

            try:
                if self.backend == "nemo":
                    self.model = self._load_nemo()
                else:
                    self.model = self._load_onnx()
            except Exception as e:
                self.log(f"[FastConformer] ERREUR de chargement : {e}")
                self.model = None
                return False

            if self.model is None:
                return False

            self.log(f"[FastConformer] Chargé en {time.perf_counter() - t0:.1f}s")
            return True

    def _load_onnx(self):
        try:
            import onnx_asr
        except ImportError:
            self.log("[FastConformer] onnx-asr non installé (pip install onnx-asr)")
            return None

        # onnx-asr ne télécharge lui-même que les dépôts qu'il connaît :
        # pour tout autre, on récupère les fichiers puis on passe le chemin local.
        path = self.model_id
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(self.model_id)

        return onnx_asr.load_model(
            ONNX_MODEL_TYPE, path, quantization=self.quantization or None,
        )

    def _load_nemo(self):
        # Erreur de configuration la plus fréquente : garder le backend nemo
        # avec un modèle ONNX. NeMo échouerait sans message exploitable.
        if "onnx" in self.model_id.lower():
            self.log(f"[FastConformer] '{self.model_id}' est un modèle ONNX : "
                     "backend 'onnx' requis. Bascule automatique.")
            self.backend = "onnx"
            return self._load_onnx()

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            self.log("[FastConformer] nemo_toolkit non installé — "
                     "bascule sur le backend ONNX. "
                     "(pip install 'nemo_toolkit[asr]')")
            self.backend = "onnx"
            self.model_id = DEFAULT_MODELS["onnx"]
            return self._load_onnx()

        # CONFLIT cuDNN — NeMo place le modèle sur GPU par défaut. Or
        # whisper_dictation importe faster_whisper au chargement du module, et
        # CTranslate2 a déjà chargé ses propres bibliothèques cuDNN : le premier
        # appel CUDA échoue alors par un crash NATIF, non rattrapable, qui tue
        # le processus sans message sous pythonw.exe.
        #   Could not load symbol cudnnGetLibConfig. Error code 127
        # map_location force le chargement sur CPU, où le modèle tourne de toute
        # façon à RTF 0,048.
        import sys

        map_location = None
        if "ctranslate2" in sys.modules or "faster_whisper" in sys.modules:
            map_location = "cpu"
            self.log("[FastConformer] CTranslate2 déjà chargé — CPU retenu "
                     "(conflit cuDNN avec CUDA).")

        if os.path.isfile(self.model_id):
            model = nemo_asr.models.ASRModel.restore_from(
                self.model_id, map_location=map_location,
            )
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_id, map_location=map_location,
            )

        if map_location == "cpu":
            model = model.to("cpu")
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _decode(self, audio):
        """Transcrit un segment unique (float32 16 kHz mono)."""
        if self.backend == "nemo":
            # L'API NeMo travaille sur des fichiers : on passe par un temporaire.
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            try:
                sf.write(path, audio, SAMPLE_RATE)
                out = self.model.transcribe([path], verbose=False)
                first = out[0] if out else ""
                # Selon la version, transcribe() renvoie des chaînes ou des objets
                return (first.text if hasattr(first, "text") else str(first)).strip()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return self.model.recognize(audio).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcrit l'audio, en le découpant si nécessaire."""
        if self.model is None and not self.load():
            raise RuntimeError("Modèle FastConformer non chargé")

        audio = np.asarray(audio, dtype=np.float32).flatten()

        if len(audio):
            clipped = float(np.mean(np.abs(audio) >= 0.999))
            if clipped > 0.02:
                self.log(f"[FastConformer] ATTENTION : {clipped:.0%} de l'audio est "
                         "saturé — baissez le gain micro.")

        chunks = self._split_audio(audio)
        if len(chunks) > 1:
            self.log(f"[FastConformer] Audio découpé en {len(chunks)} segments")

        parts = []
        with self._lock:
            for chunk in chunks:
                if len(chunk) < SAMPLE_RATE * 0.1:
                    continue
                text = self._decode(chunk)
                if text:
                    parts.append(text)

        return " ".join(parts).strip()

    def _split_audio(self, audio):
        """Découpe l'audio sur les silences si sa durée dépasse la limite."""
        max_samples = int(MAX_CHUNK_SECONDS * SAMPLE_RATE)
        if len(audio) <= max_samples:
            return [audio]

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + max_samples, len(audio))

            if end < len(audio):
                search_start = start + int(max_samples * 0.75)
                window = np.abs(audio[search_start:end])
                frame = 400  # 25 ms
                n_frames = len(window) // frame
                if n_frames > 0:
                    energies = window[:n_frames * frame].reshape(n_frames, frame).mean(axis=1)
                    end = search_start + int(np.argmin(energies)) * frame

            chunks.append(audio[start:end])
            start = end

        return chunks

    def unload(self):
        """Libère le modèle."""
        with self._lock:
            self.model = None


# =============================================================================
# Singleton pour l'application
# =============================================================================
_engine = None
_engine_lock = threading.Lock()


def get_engine(config=None, log=None):
    """Retourne le moteur FastConformer partagé, en le créant au besoin."""
    global _engine

    config = config or {}
    backend = config.get("fastconformer_backend", "onnx")
    if backend not in DEFAULT_MODELS:
        backend = "onnx"
    # Champ vide -> le modèle par défaut du backend choisi, pas un format
    # que ce backend serait incapable de charger.
    model_id = config.get("fastconformer_model", "").strip() or DEFAULT_MODELS[backend]
    quantization = config.get("fastconformer_quantization", "int8")

    # Chemin relatif -> racine du projet (le lanceur s'élève en administrateur,
    # ce qui déplace le répertoire courant hors du projet).
    if not os.path.isabs(model_id) and not os.path.isdir(model_id):
        candidate = os.path.join(BASE_DIR, model_id)
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            model_id = candidate

    with _engine_lock:
        if (_engine is not None
                and _engine.model_id == model_id
                and _engine.backend == backend):
            return _engine

        if _engine is not None:
            _engine.unload()

        _engine = FastConformerEngine(
            model_id=model_id, backend=backend,
            quantization=quantization, log=log,
        )
        return _engine


def transcribe(audio: np.ndarray, config=None, log=None) -> str:
    """Raccourci : transcrit l'audio avec le moteur partagé."""
    return get_engine(config, log).transcribe(audio)
