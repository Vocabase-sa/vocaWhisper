"""
Banc d'essai multi-moteurs ASR — cible CPU / streaming / français.
===================================================================

Compare sur le corpus du projet les moteurs candidats pour un déploiement
SANS GPU : WER, WER après correction fuzzy, taux de reconnaissance des noms
propres, latence et RTF.

Le RTF (Real-Time Factor) est le critère décisif pour le streaming : en dessous
de 1,0 le moteur transcrit plus vite que le temps réel, donc il tient un flux
continu. Au-delà, il décroche.

Moteurs supportés (syntaxe --engines) :
    whisper:<taille>[:<compute>]   faster-whisper, ex. whisper:small:int8
    whisper:<chemin_ct2>           modèle CTranslate2 local (fine-tuné)
    wav2vec2:<model_id>            CTC, ex. wav2vec2:bofenghuang/asr-wav2vec2-ctc-french
    moonshine:<model_id|chemin>    via moonshine_engine
    vosk:<chemin_modele>           streaming natif (pip install vosk)

Usage :
    python utils/bench_engines.py --limit 50
    python utils/bench_engines.py --engines whisper:base:int8 whisper:small:int8
    python utils/bench_engines.py --engines vosk:models/vosk-model-fr-0.22

Tous les moteurs tournent sur CPU par défaut : c'est le point de la comparaison.
"""

import argparse
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

from test_moonshine import (  # noqa: E402
    SAMPLE_RATE,
    apply_fuzzy,
    build_directory,
    load_samples,
    score,
)

DEFAULT_ENGINES = [
    "whisper:base:int8",
    "whisper:small:int8",
    "moonshine:",
]


# =============================================================================
# Adaptateurs de moteurs
# =============================================================================
class WhisperEngine:
    """faster-whisper sur CPU (CTranslate2)."""

    def __init__(self, spec):
        parts = spec.split(":", 1)[1].split(":")
        self.model_name = parts[0]
        self.compute = parts[1] if len(parts) > 1 else "int8"
        self.label = f"whisper:{self.model_name}:{self.compute}"
        self.model = None

    def load(self):
        from faster_whisper import WhisperModel

        path = self.model_name
        if not os.path.isabs(path) and os.path.isdir(os.path.join(BASE_DIR, path)):
            path = os.path.join(BASE_DIR, path)
        self.model = WhisperModel(path, device="cpu", compute_type=self.compute)

        # Même conditionnement que l'application
        self.prompt = None
        vocab = os.path.join(BASE_DIR, "vocabulaire.txt")
        if os.path.isfile(vocab):
            with open(vocab, encoding="utf-8") as f:
                self.prompt = " ".join(
                    l.strip() for l in f if l.strip() and not l.startswith("#")
                ) or None

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="fr", beam_size=5, vad_filter=False,
            initial_prompt=self.prompt,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class Wav2Vec2Engine:
    """wav2vec2 CTC — pas d'hallucination possible, découpage trivial."""

    def __init__(self, spec):
        self.model_id = spec.split(":", 1)[1]
        self.label = f"wav2vec2:{self.model_id.split('/')[-1]}"
        self.model = None

    def load(self):
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2Processor

        self.torch = torch
        # Wav2Vec2Processor explicitement, et non AutoProcessor : plusieurs
        # modèles FR publient un Wav2Vec2ProcessorWithLM, qui exige pyctcdecode
        # et un modèle de langage n-gram. On s'en tient au décodage CTC simple.
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id)
        self.model.eval()

    def transcribe(self, audio):
        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        ids = self.torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0].strip()


class MoonshineAdapter:
    """Moonshine via le moteur du projet."""

    def __init__(self, spec):
        model_id = spec.split(":", 1)[1].strip()
        self.model_id = model_id or None
        self.label = f"moonshine:{(model_id or 'tiny-fr').split('/')[-1]}"
        self.engine = None

    def load(self):
        from moonshine_engine import MoonshineEngine

        self.engine = MoonshineEngine(model_id=self.model_id, device="cpu")
        if not self.engine.load():
            raise RuntimeError("chargement Moonshine impossible")

    def transcribe(self, audio):
        return self.engine.transcribe(audio)


class VoskEngine:
    """Vosk / Kaldi — streaming natif, vocabulaire reconfigurable."""

    def __init__(self, spec):
        self.model_path = spec.split(":", 1)[1]
        self.label = f"vosk:{os.path.basename(self.model_path.rstrip('/'))}"
        self.model = None

    def load(self):
        import json

        import vosk

        self.json = json
        self.vosk = vosk
        vosk.SetLogLevel(-1)

        path = self.model_path
        if not os.path.isabs(path) and os.path.isdir(os.path.join(BASE_DIR, path)):
            path = os.path.join(BASE_DIR, path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"modèle Vosk introuvable : {path}")
        self.model = vosk.Model(path)

    def transcribe(self, audio):
        rec = self.vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16).tobytes()
        rec.AcceptWaveform(pcm)
        return self.json.loads(rec.FinalResult()).get("text", "").strip()


def record_microphone(seconds, gain=1.0):
    """Enregistre depuis le micro configuré dans l'application.

    Reprend les réglages de whisper_dictation (16 kHz mono, même périphérique)
    pour que la comparaison porte sur des conditions réelles de dictée.
    """
    import sounddevice as sd

    device = None
    try:
        from whisper_dictation import get_microphone_device
        device = get_microphone_device()
    except Exception:
        pass

    print(f"\n  Enregistrement de {seconds}s — parlez maintenant...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=device)
    sd.wait()
    audio = audio.flatten()

    peak = float(np.abs(audio).max())
    if gain != 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
        clipped = float(np.mean(np.abs(audio) >= 0.999))
        print(f"  Terminé — pic {peak:.3f}, gain x{gain}, saturation {clipped:.1%}")
        if clipped > 0.02:
            print("  ATTENTION : audio saturé, baissez le gain.")
    else:
        print(f"  Terminé — pic {peak:.3f}")

    return audio


def load_wav(path):
    """Charge un fichier audio quelconque en float32 mono 16 kHz."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def compare_single(audio, specs, reference=None):
    """Transcrit un seul extrait avec tous les moteurs et affiche le comparatif."""
    duration = len(audio) / SAMPLE_RATE
    print(f"\n  Audio : {duration:.2f}s\n")
    print("=" * 78)
    print(f"  {'moteur':<34} {'temps':>8} {'RTF':>6}  transcription")
    print("  " + "-" * 74)

    for spec in specs:
        try:
            engine = build_engine(spec)
            engine.load()
        except Exception as e:
            print(f"  {spec:<34} {'ECHEC':>8}        {type(e).__name__}: {str(e)[:60]}")
            continue

        t0 = time.perf_counter()
        try:
            hyp = engine.transcribe(audio)
        except Exception as e:
            hyp = f"<erreur: {type(e).__name__}>"
        elapsed = time.perf_counter() - t0

        print(f"  {engine.label:<34} {elapsed:>7.2f}s {elapsed / duration:>6.2f}  {hyp}")

    if reference:
        print("  " + "-" * 74)
        print(f"  {'référence':<34} {'':>8} {'':>6}  {reference}")
    print()


class FastConformerAdapter:
    """FastConformer NeMo, via le moteur du projet (backend onnx ou nemo)."""

    def __init__(self, spec):
        rest = spec.split(":", 1)[1].strip()
        # Forme "fastconformer:nemo:<model_id>" pour forcer le backend NeMo
        if rest.startswith("nemo:"):
            self.backend, model_id = "nemo", rest[5:]
        elif rest.startswith("onnx:"):
            self.backend, model_id = "onnx", rest[5:]
        else:
            self.backend, model_id = "onnx", rest

        self.model_id = model_id or None
        short = (model_id or "openvoiceos").split("/")[-1][:28]
        self.label = f"fastconf:{self.backend}:{short}"
        self.engine = None

    def load(self):
        from fastconformer_engine import FastConformerEngine

        self.engine = FastConformerEngine(
            model_id=self.model_id, backend=self.backend,
        )
        if not self.engine.load():
            raise RuntimeError("chargement FastConformer impossible")

    def transcribe(self, audio):
        return self.engine.transcribe(audio)


def build_engine(spec):
    """Instancie l'adaptateur correspondant à la spécification."""
    if spec.startswith("whisper:"):
        return WhisperEngine(spec)
    if spec.startswith("wav2vec2:"):
        return Wav2Vec2Engine(spec)
    if spec.startswith("moonshine:"):
        return MoonshineAdapter(spec)
    if spec.startswith("fastconformer:"):
        return FastConformerAdapter(spec)
    if spec.startswith("vosk:"):
        return VoskEngine(spec)
    raise ValueError(f"moteur inconnu : {spec}")


# =============================================================================
# Exécution
# =============================================================================
def run_engine(spec, samples):
    """Charge un moteur, transcrit tous les échantillons, mesure les temps."""
    engine = build_engine(spec)
    print(f"\n  [{engine.label}] chargement...")

    t0 = time.perf_counter()
    try:
        engine.load()
    except Exception as e:
        print(f"    ECHEC : {type(e).__name__}: {str(e)[:160]}")
        return None
    load_time = time.perf_counter() - t0
    print(f"    chargé en {load_time:.1f}s — transcription...")

    results = []
    for i, (audio, _) in enumerate(samples, 1):
        t = time.perf_counter()
        try:
            hyp = engine.transcribe(audio)
        except Exception as e:
            print(f"    ERREUR sur l'échantillon {i}: {type(e).__name__}: {str(e)[:100]}")
            hyp = ""
        results.append((hyp, time.perf_counter() - t, len(audio) / SAMPLE_RATE))
        if i % 25 == 0:
            print(f"      {i}/{len(samples)}...")

    return {"label": engine.label, "results": results, "load_time": load_time}


def main():
    parser = argparse.ArgumentParser(description="Banc d'essai multi-moteurs ASR (CPU)")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES,
                        help="Moteurs à comparer (voir la docstring du module)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Nombre d'échantillons (défaut: 50)")
    parser.add_argument("--dataset",
                        default=os.path.join(BASE_DIR, "fine_tuning", "dataset_prepared"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv",
                        default=os.path.join(BASE_DIR, "fine_tuning", "data", "transcriptions.csv"))
    parser.add_argument("--audio_dir",
                        default=os.path.join(BASE_DIR, "fine_tuning", "data", "audio"))
    parser.add_argument("--fuzzy_threshold", type=int, default=65)
    parser.add_argument("--show", type=int, default=5,
                        help="Exemples détaillés par moteur (défaut: 5)")
    parser.add_argument("--mic", type=float, metavar="SECONDES", default=None,
                        help="Enregistrer au micro et comparer les moteurs sur cet extrait")
    parser.add_argument("--wav", type=str, default=None,
                        help="Comparer les moteurs sur un fichier audio donné")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Gain appliqué à l'enregistrement micro (défaut: 1.0)")
    parser.add_argument("--ref", type=str, default=None,
                        help="Transcription attendue, affichée pour comparaison")
    args = parser.parse_args()

    print("=" * 78)
    print("  Banc d'essai multi-moteurs ASR — cible CPU")
    print("=" * 78)

    # --- Modes extrait unique : micro ou fichier ---
    if args.mic or args.wav:
        audio = (record_microphone(args.mic, args.gain) if args.mic
                 else load_wav(args.wav))
        compare_single(audio, args.engines, args.ref)
        return

    print()
    samples = load_samples(args)
    if not samples:
        print("[ERREUR] aucun échantillon chargé")
        sys.exit(1)

    total = sum(len(a) for a, _ in samples) / SAMPLE_RATE
    print(f"  {len(samples)} échantillons — {total / 60:.1f} min d'audio")

    directory = build_directory(samples)
    print(f"  Annuaire fuzzy : {len(directory)} noms (seuil {args.fuzzy_threshold})")

    runs = [r for r in (run_engine(s, samples) for s in args.engines) if r]
    if not runs:
        print("\n[ERREUR] aucun moteur n'a pu être évalué")
        sys.exit(1)

    # --- Tableau récapitulatif ---
    print("\n" + "=" * 78)
    print("  RÉSULTATS (CPU)")
    print("=" * 78)
    header = f"\n  {'moteur':<34} {'WER':>7} {'WER+fz':>7} {'noms+fz':>8} {'lat.':>8} {'RTF':>6}"
    print(header)
    print("  " + "-" * 74)

    rows = []
    for run in runs:
        s = score(samples, run["results"])
        sf_ = score(samples, apply_fuzzy(run["results"], directory, args.fuzzy_threshold))
        rows.append((run, s, sf_))
        print(f"  {run['label']:<34} {s['wer']:>6.1f}% {sf_['wer']:>6.1f}% "
              f"{sf_['nouns_pct']:>7.1f}% {s['latency_ms']:>6.0f}ms {s['rtf']:>6.3f}")

    print("\n  RTF < 1,0 = plus rapide que le temps réel (streaming tenable sur CPU)")

    # --- Exemples ---
    if args.show:
        print("\n" + "=" * 78)
        print(f"  EXEMPLES ({min(args.show, len(samples))} premiers)")
        print("=" * 78)
        for i in range(min(args.show, len(samples))):
            print(f"\n  réf : {samples[i][1]}")
            for run, _, _ in rows:
                print(f"    {run['label']:<32} {run['results'][i][0]}")

    print()


if __name__ == "__main__":
    main()
