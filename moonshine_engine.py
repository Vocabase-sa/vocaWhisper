"""
Moteur de transcription Moonshine (ASR edge, faible latence).
=============================================================

Moonshine est une alternative à Whisper conçue pour le temps réel : son encodeur
accepte des fenêtres audio de longueur variable, là où Whisper padde tout à 30 s.
Sur un énoncé de 2 s (« Je voudrais le docteur X »), le coût de calcul est
proportionnel à la durée réelle — d'où un intérêt marqué pour la voie RTP.

Deux backends :
    - torch (défaut) : transformers, fonctionne sur CUDA et CPU
    - onnx           : onnxruntime via optimum, plus rapide sur CPU

LIMITE IMPORTANTE — pas d'initial_prompt :
    Contrairement à faster-whisper, Moonshine n'accepte pas de prompt de
    conditionnement. Le contenu de vocabulaire.txt ne peut donc PAS être injecté
    pour biaiser la reconnaissance. Les corrections restent appliquées en aval
    (corrections.txt + fuzzy_correction), mais le biasing amont est perdu.
    C'est la contrepartie à mettre en balance avec le gain de latence.

Modèles :
    Les modèles Moonshine officiels (v1 « Flavors », v2 streaming) ne couvrent
    PAS le français. Le défaut pointe donc sur le fine-tune communautaire
    Cornebidouil/moonshine-tiny-fr (27M, ~21,8 % WER sur MLS-French), à
    re-spécialiser sur ton corpus via fine_tuning/train_moonshine.py.
"""

import os
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_MODEL = "Cornebidouil/moonshine-tiny-fr"

# Au-delà de cette durée, l'audio est découpé : Moonshine v1 se dégrade sur les
# séquences longues (entraîné sur des segments courts).
MAX_CHUNK_SECONDS = 30.0


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Au-delà de cette proportion d'échantillons saturés, la transcription se
# dégrade nettement sur un modèle de cette taille : on alerte.
CLIPPING_WARN_RATIO = 0.02


def _default_log(msg):
    print(msg)


def resolve_model_path(model_id: str) -> str:
    """Résout un chemin de modèle relatif contre la racine du projet.

    run_silent.vbs s'auto-élève en administrateur, ce qui place le répertoire
    courant du processus dans C:\\Windows\\System32. Un chemin relatif comme
    "fine_tuning/output_moonshine/final" y est introuvable : il serait alors
    transmis tel quel à from_pretrained(), interprété comme un identifiant
    Hugging Face, et le chargement échouerait sans modèle de secours.

    Un identifiant Hugging Face ("Cornebidouil/moonshine-tiny-fr") est laissé
    intact : il ne correspond à aucun dossier local.
    """
    if not model_id or os.path.isabs(model_id) or os.path.isdir(model_id):
        return model_id

    candidate = os.path.join(BASE_DIR, model_id)
    return candidate if os.path.isdir(candidate) else model_id


class MoonshineEngine:
    """Encapsule le chargement et l'inférence d'un modèle Moonshine."""

    def __init__(self, model_id=None, device="auto", backend="torch", log=None):
        self.model_id = model_id or DEFAULT_MODEL
        self.backend = backend
        self.log = log or _default_log
        self.model = None
        self.processor = None
        self._lock = threading.Lock()
        self._device = self._resolve_device(device, self.log)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device, log=_default_log):
        """Résout 'auto' vers le meilleur device utilisable.

        CONFLIT cuDNN — pourquoi 'auto' évite CUDA ici :
            faster-whisper embarque CTranslate2, qui charge ses propres
            bibliothèques cuDNN. Si torch charge ensuite les siennes dans le
            même processus, le chargement échoue au premier appel CUDA :
                Could not load symbol cudnnGetLibConfig. Error code 127
            C'est un crash NATIF, pas une exception Python : impossible à
            rattraper, et sous pythonw.exe le processus meurt sans message.

            whisper_dictation.py importe faster_whisper au chargement du module,
            donc le conflit est systématique dès que Moonshine passe sur CUDA.

            Le coût est nul : mesuré sur ce corpus, Moonshine sur CPU (212 ms)
            est plus rapide que faster-whisper large-v3 sur RTX 4090 (372 ms),
            et laisse le GPU entièrement libre.

            Un device explicite ('cuda') reste respecté — à n'utiliser que dans
            un processus où faster-whisper n'est pas chargé.
        """
        if device and device != "auto":
            return device

        if "ctranslate2" in sys.modules or "faster_whisper" in sys.modules:
            log("[Moonshine] CTranslate2 déjà chargé — CPU retenu "
                "(conflit cuDNN avec CUDA).")
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def device(self):
        return self._device

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def load(self):
        """Charge le modèle et le processeur. Idempotent."""
        if self.model is not None:
            return True

        with self._lock:
            if self.model is not None:
                return True

            t0 = time.perf_counter()
            self.log(f"[Moonshine] Chargement de {self.model_id} ({self.backend}, {self._device})...")

            try:
                from transformers import AutoProcessor
            except ImportError:
                self.log("[Moonshine] ERREUR : transformers non installé "
                         "(pip install -r requirements-moonshine.txt)")
                return False

            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)

                if self.backend == "onnx":
                    self.model = self._load_onnx()
                else:
                    self.model = self._load_torch()

                if self.model is None:
                    return False

            except Exception as e:
                self.log(f"[Moonshine] ERREUR de chargement : {e}")
                self.model = None
                self.processor = None
                return False

            elapsed = time.perf_counter() - t0
            self.log(f"[Moonshine] Modèle chargé en {elapsed:.1f}s")
            self._warmup()
            return True

    def _load_torch(self):
        from transformers import MoonshineForConditionalGeneration

        model = MoonshineForConditionalGeneration.from_pretrained(self.model_id)
        model.to(self._device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        self.log(f"[Moonshine] {n_params:.1f}M paramètres")
        return model

    def _load_onnx(self):
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        except ImportError:
            self.log("[Moonshine] optimum non installé — bascule sur le backend torch. "
                     "(pip install 'optimum[onnxruntime]')")
            self.backend = "torch"
            return self._load_torch()

        provider = "CUDAExecutionProvider" if self._device == "cuda" else "CPUExecutionProvider"
        return ORTModelForSpeechSeq2Seq.from_pretrained(self.model_id, provider=provider)

    def _warmup(self):
        """Préchauffe le modèle pour éliminer la latence du premier appel."""
        try:
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            self._generate(silence)
            self.log("[Moonshine] Préchauffage terminé")
        except Exception as e:
            self.log(f"[Moonshine] Préchauffage ignoré : {e}")

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _generate(self, audio):
        """Transcrit un segment audio unique (float32 16 kHz mono)."""
        import torch

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        if self.backend == "torch":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Budget de tokens proportionnel à la durée : ~6 tokens/seconde de parole,
        # avec un plancher pour les énoncés très courts.
        duration = len(audio) / SAMPLE_RATE
        max_new_tokens = max(16, min(256, int(duration * 6) + 16))

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcrit l'audio, en le découpant si nécessaire.

        Args:
            audio: forme d'onde float32 mono à 16 kHz, dans [-1, 1].

        Returns:
            Le texte transcrit (chaîne vide si rien n'est détecté).
        """
        if self.model is None and not self.load():
            raise RuntimeError("Modèle Moonshine non chargé")

        audio = np.asarray(audio, dtype=np.float32).flatten()

        # Le gain micro de l'application est appliqué en amont (stop_recording),
        # puis l'audio est écrêté à [-1, 1]. Whisper encaisse cet écrêtage ;
        # Moonshine, entraîné sur de l'audio non saturé et 50x plus petit, le
        # transcrit mal. L'information détruite par le clipping est perdue :
        # on ne peut que le signaler.
        if len(audio):
            clipped = float(np.mean(np.abs(audio) >= 0.999))
            if clipped > CLIPPING_WARN_RATIO:
                self.log(
                    f"[Moonshine] ATTENTION : {clipped:.0%} de l'audio est saturé — "
                    "baissez le gain micro (2-3 conseillé avec Moonshine)."
                )

        chunks = self._split_audio(audio)
        if len(chunks) > 1:
            self.log(f"[Moonshine] Audio découpé en {len(chunks)} segments")

        parts = []
        with self._lock:
            for chunk in chunks:
                text = self._generate(chunk)
                if text:
                    parts.append(text)

        return " ".join(parts).strip()

    # ------------------------------------------------------------------
    # Découpage des audios longs
    # ------------------------------------------------------------------
    def _split_audio(self, audio):
        """Découpe l'audio sur les silences si sa durée dépasse la limite."""
        max_samples = int(MAX_CHUNK_SECONDS * SAMPLE_RATE)
        if len(audio) <= max_samples:
            return [audio]

        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + max_samples, len(audio))

            # Chercher un creux d'énergie dans le dernier quart pour couper
            # entre deux mots plutôt qu'au milieu d'un.
            if end < len(audio):
                search_start = start + int(max_samples * 0.75)
                window = np.abs(audio[search_start:end])
                if len(window) > 0:
                    frame = 400  # 25 ms
                    n_frames = len(window) // frame
                    if n_frames > 0:
                        energies = window[:n_frames * frame].reshape(n_frames, frame).mean(axis=1)
                        end = search_start + int(np.argmin(energies)) * frame

            chunks.append(audio[start:end])
            start = end

        return chunks

    def unload(self):
        """Libère le modèle et la VRAM associée."""
        with self._lock:
            self.model = None
            self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# =============================================================================
# Singleton pour l'application
# =============================================================================
_engine = None
_engine_lock = threading.Lock()


def get_engine(config=None, log=None):
    """Retourne le moteur Moonshine partagé, en le créant au besoin.

    Le moteur est recréé si la configuration a changé (modèle ou backend).
    """
    global _engine

    config = config or {}
    model_id = config.get("moonshine_model", "").strip() or DEFAULT_MODEL
    backend = config.get("moonshine_backend", "torch")
    device = config.get("moonshine_device", "auto")

    model_id = resolve_model_path(model_id)

    with _engine_lock:
        if (_engine is not None
                and _engine.model_id == model_id
                and _engine.backend == backend):
            return _engine

        if _engine is not None:
            _engine.unload()

        _engine = MoonshineEngine(
            model_id=model_id,
            device=device,
            backend=backend,
            log=log,
        )
        return _engine


def transcribe(audio: np.ndarray, config=None, log=None) -> str:
    """Raccourci : transcrit l'audio avec le moteur partagé."""
    return get_engine(config, log).transcribe(audio)
