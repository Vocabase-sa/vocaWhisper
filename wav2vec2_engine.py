"""
Moteur de transcription wav2vec2 (CTC, français).
==================================================

Troisième famille testable dans VocaWhisper, à côté de faster-whisper et de
Moonshine. wav2vec2 est un modèle CTC : il classe chaque trame audio en
caractère, sans décodeur autorégressif.

Ce que ça change en pratique :
    - PAS d'hallucination possible. Un modèle Whisper peut inventer une phrase
      entière sur du silence ; un CTC ne produit que ce qu'il entend.
    - Sortie en minuscules, SANS ponctuation ni majuscules. C'est normal :
      le modèle prédit des caractères, pas de la mise en forme.
    - Découpage trivial : aucune dépendance entre les trames, donc n'importe
      quel segment se transcrit indépendamment.

LIMITE — pas d'initial_prompt :
    Comme Moonshine, wav2vec2 n'accepte aucun conditionnement. vocabulaire.txt
    n'est pas injecté ; corrections.txt et le fuzzy s'appliquent en aval.

Modèle par défaut : jonatasgrosman/wav2vec2-large-xlsr-53-french (315M, ~1,2 Go).
Prévoir un premier téléchargement d'environ 30 s.
"""

import os
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOMS_PROPRES_FILE = os.path.join(BASE_DIR, "noms_propres.txt")

# Poids du boosting. Mesuré sur le corpus du projet (60 extraits, annuaire de
# 1203 noms) : le taux de noms propres passe de 23,8 % à 50,8 % à poids 10.
# Au-delà, le décodeur force des noms là où il n'y en a pas et le WER global se
# dégrade (29,6 % à poids 20, 32,8 % à poids 40).
DEFAULT_HOTWORD_WEIGHT = 10.0

# Mesuré sur CPU (32 cœurs) : le RTF reste constant à ~0,06 de 5 s à 40 s, donc
# le coût est linéaire sur cette plage — l'attention quadratique ne domine pas
# encore. On découpe néanmoins au-delà de 30 s pour borner la mémoire.
MAX_CHUNK_SECONDS = 30.0


def _default_log(msg):
    print(msg)


def resolve_model_path(model_id: str) -> str:
    """Résout un chemin relatif contre la racine du projet.

    Même motif que moonshine_engine : le lanceur s'élève en administrateur,
    ce qui déplace le répertoire courant hors du projet.
    """
    if not model_id or os.path.isabs(model_id) or os.path.isdir(model_id):
        return model_id

    candidate = os.path.join(BASE_DIR, model_id)
    return candidate if os.path.isdir(candidate) else model_id


class Wav2Vec2Engine:
    """Encapsule le chargement et l'inférence d'un modèle wav2vec2 CTC."""

    def __init__(self, model_id=None, device="auto", log=None,
                 hotwords_enabled=False, hotword_weight=DEFAULT_HOTWORD_WEIGHT):
        self.model_id = model_id or DEFAULT_MODEL
        self.log = log or _default_log
        self.model = None
        self.processor = None
        self.hotwords_enabled = hotwords_enabled
        self.hotword_weight = hotword_weight
        self.decoder = None          # BeamSearchDecoderCTC, si hotwords actifs
        self.hotwords = []
        self._lock = threading.Lock()
        self._device = self._resolve_device(device, self.log)

    @staticmethod
    def _resolve_device(device, log=_default_log):
        """Résout 'auto', en évitant le conflit cuDNN avec CTranslate2.

        Voir moonshine_engine._resolve_device : faster-whisper charge ses
        propres bibliothèques cuDNN, et torch sur CUDA dans le même processus
        provoque un crash natif non rattrapable.
        """
        import sys

        if device and device != "auto":
            return device

        if "ctranslate2" in sys.modules or "faster_whisper" in sys.modules:
            log("[wav2vec2] CTranslate2 déjà chargé — CPU retenu "
                "(conflit cuDNN avec CUDA).")
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
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
            self.log(f"[wav2vec2] Chargement de {self.model_id} ({self._device})...")

            try:
                from transformers import AutoModelForCTC, Wav2Vec2Processor
            except ImportError:
                self.log("[wav2vec2] ERREUR : transformers non installé "
                         "(pip install -r requirements-moonshine.txt)")
                return False

            try:
                # Wav2Vec2Processor explicitement : plusieurs modèles FR publient
                # un Wav2Vec2ProcessorWithLM, qui exigerait pyctcdecode et un
                # modèle de langage n-gram. On s'en tient au CTC simple.
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
                self.model = AutoModelForCTC.from_pretrained(self.model_id)
                self.model.to(self._device)
                self.model.eval()
            except Exception as e:
                self.log(f"[wav2vec2] ERREUR de chargement : {e}")
                self.model = None
                self.processor = None
                return False

            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            self.log(f"[wav2vec2] {n_params:.0f}M paramètres — "
                     f"chargé en {time.perf_counter() - t0:.1f}s")

            if self.hotwords_enabled:
                self._build_decoder()

            return True

    # ------------------------------------------------------------------
    # Décodage par recherche en faisceau avec hotwords
    # ------------------------------------------------------------------
    def _load_hotwords(self):
        """Charge l'annuaire depuis noms_propres.txt, normalisé pour le CTC.

        Le vocabulaire CTC ne connaît que les minuscules : un hotword en
        majuscules ne correspondrait à aucune sortie possible du modèle.
        """
        if not os.path.isfile(NOMS_PROPRES_FILE):
            return []

        words = []
        with open(NOMS_PROPRES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    words.append(line.lower())
        return words

    def _build_decoder(self):
        """Construit le décodeur pyctcdecode à partir du vocabulaire du modèle."""
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            self.log("[wav2vec2] pyctcdecode non installé — hotwords ignorés. "
                     "(pip install pyctcdecode pygtrie)")
            self.hotwords_enabled = False
            return

        self.hotwords = self._load_hotwords()
        if not self.hotwords:
            self.log(f"[wav2vec2] Aucun nom dans {os.path.basename(NOMS_PROPRES_FILE)} — "
                     "hotwords sans effet.")
            self.hotwords_enabled = False
            return

        # pyctcdecode attend les labels ordonnés par index, SANS doublon :
        # une seule chaîne vide (le blank CTC), un espace pour le délimiteur de
        # mot, et des marqueurs uniques pour les tokens spéciaux jamais émis.
        #
        # La taille vient du MODÈLE, pas du tokenizer : plusieurs modèles
        # français déclarent des tokens (<s>, </s>) auxquels la tête CTC ne
        # correspond pas. bofenghuang expose ainsi 52 tokens pour 50 sorties, et
        # pyctcdecode refuse alors les logits pour cause de dimension.
        vocab = self.processor.tokenizer.get_vocab()
        n_labels = getattr(self.model.config, "vocab_size", None) or (max(vocab.values()) + 1)

        raw = [None] * n_labels
        for token, index in vocab.items():
            if index < n_labels:
                raw[index] = token

        pad_token = self.processor.tokenizer.pad_token
        labels, filler = [], 0
        for token in raw:
            if token == pad_token:
                labels.append("")
            elif token == "|":
                labels.append(" ")
            elif token is None or token in ("<s>", "</s>", "<unk>"):
                # Indice sans token déclaré, ou token spécial : marqueur unique
                # que le décodeur n'émettra jamais.
                filler += 1
                labels.append("⁇" * filler)
            else:
                labels.append(token)

        self.decoder = build_ctcdecoder(labels)
        self.log(f"[wav2vec2] Hotwords actifs : {len(self.hotwords)} noms "
                 f"(poids {self.hotword_weight})")

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _decode(self, audio):
        """Transcrit un segment unique (float32 16 kHz mono)."""
        import torch

        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Recherche en faisceau si les hotwords sont actifs : elle explore
        # plusieurs hypothèses et privilégie celles contenant un nom de
        # l'annuaire. Coûte ~300 ms par énoncé contre ~0 en glouton.
        if self.decoder is not None:
            return self.decoder.decode(
                logits[0].cpu().numpy(),
                hotwords=self.hotwords,
                hotword_weight=self.hotword_weight,
            ).strip()

        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0].strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcrit l'audio, en le découpant si nécessaire."""
        if self.model is None and not self.load():
            raise RuntimeError("Modèle wav2vec2 non chargé")

        audio = np.asarray(audio, dtype=np.float32).flatten()

        if len(audio):
            clipped = float(np.mean(np.abs(audio) >= 0.999))
            if clipped > 0.02:
                self.log(f"[wav2vec2] ATTENTION : {clipped:.0%} de l'audio est "
                         "saturé — baissez le gain micro.")

        chunks = self._split_audio(audio)
        if len(chunks) > 1:
            self.log(f"[wav2vec2] Audio découpé en {len(chunks)} segments")

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
        """Découpe l'audio sur les silences si sa durée dépasse la limite.

        Couper au milieu d'un mot coûte cher en CTC : les deux moitiés donnent
        chacune une suite de caractères erronée. On cherche donc un creux
        d'énergie plutôt que de trancher à date fixe.
        """
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
    """Retourne le moteur wav2vec2 partagé, en le créant au besoin."""
    global _engine

    config = config or {}
    model_id = resolve_model_path(
        config.get("wav2vec2_model", "").strip() or DEFAULT_MODEL
    )
    device = config.get("wav2vec2_device", "auto")
    hotwords_enabled = config.get("wav2vec2_hotwords", False)
    hotword_weight = float(config.get("wav2vec2_hotword_weight", DEFAULT_HOTWORD_WEIGHT))

    with _engine_lock:
        if (_engine is not None
                and _engine.model_id == model_id
                and _engine.hotwords_enabled == hotwords_enabled
                and _engine.hotword_weight == hotword_weight):
            return _engine

        if _engine is not None:
            _engine.unload()

        _engine = Wav2Vec2Engine(
            model_id=model_id, device=device, log=log,
            hotwords_enabled=hotwords_enabled, hotword_weight=hotword_weight,
        )
        return _engine


def transcribe(audio: np.ndarray, config=None, log=None) -> str:
    """Raccourci : transcrit l'audio avec le moteur partagé."""
    return get_engine(config, log).transcribe(audio)
