"""
Banc d'essai Moonshine vs Whisper sur le corpus du projet.
===========================================================

Compare, sur les mêmes fichiers audio :
    - le WER global
    - le taux de reconnaissance des NOMS PROPRES (la vraie métrique métier :
      un standard téléphonique qui comprend « je voudrais le docteur » mais
      rate le nom ne sert à rien)
    - la latence par énoncé

Les noms propres sont détectés comme les tokens écrits en MAJUSCULES dans la
transcription de référence (convention du corpus : « Je voudrais le docteur
ABOMULAY. »).

Usage :
    python utils/test_moonshine.py
    python utils/test_moonshine.py --limit 100
    python utils/test_moonshine.py --moonshine_model fine_tuning/output_moonshine/final
    python utils/test_moonshine.py --no_whisper          # Moonshine seul
    python utils/test_moonshine.py --moonshine_device cpu

Prérequis :
    pip install -r requirements-moonshine.txt
"""

import argparse
import os
import re
import sys
import time
import unicodedata

import numpy as np

# Permettre l'import des modules à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DEFAULT_DATASET = os.path.join(BASE_DIR, "fine_tuning", "dataset_prepared")
DEFAULT_CSV = os.path.join(BASE_DIR, "fine_tuning", "data", "transcriptions.csv")
DEFAULT_AUDIO_DIR = os.path.join(BASE_DIR, "fine_tuning", "data", "audio")
DEFAULT_CT2 = os.path.join(BASE_DIR, "fine_tuning", "model_ct2")

SAMPLE_RATE = 16000


# =============================================================================
# Normalisation et métriques
# =============================================================================
def normalize(text: str) -> str:
    """Minuscules, sans accents ni ponctuation — pour comparer équitablement."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_proper_nouns(reference: str) -> list[str]:
    """Extrait les tokens en MAJUSCULES : la convention du corpus pour les noms."""
    return [
        normalize(tok)
        for tok in re.findall(r"\b[A-ZÀ-Ý][A-ZÀ-Ý'\-]{2,}\b", reference)
    ]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER par distance de Levenshtein sur les mots."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()

    if not ref:
        return 0.0 if not hyp else 1.0

    # Programmation dynamique sur deux lignes
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr

    return prev[-1] / len(ref)


# =============================================================================
# Chargement des échantillons
# =============================================================================
def load_samples(args):
    """Charge les paires (audio, référence) depuis le dataset ou le CSV."""
    samples = []

    if os.path.isdir(args.dataset):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(args.dataset)
            split = args.split if args.split in ds else list(ds.keys())[0]
            ds = ds[split]
            print(f"  Source : dataset_prepared (split '{split}', {len(ds)} exemples)")

            for row in ds:
                audio = np.asarray(row["audio"]["array"], dtype=np.float32)
                samples.append((audio, row["sentence"]))
                if args.limit and len(samples) >= args.limit:
                    break
            return samples
        except Exception as e:
            print(f"  [WARN] Lecture du dataset impossible ({e}), bascule sur le CSV")

    # Repli : CSV + dossier audio
    import csv
    import soundfile as sf

    if not os.path.isfile(args.csv):
        print(f"\n[ERREUR] Ni dataset ni CSV trouvés ({args.csv})")
        sys.exit(1)

    print(f"  Source : {args.csv}")
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(args.audio_dir, row["audio_file"])
            if not os.path.isfile(path):
                continue
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            samples.append((audio, row["transcription"]))
            if args.limit and len(samples) >= args.limit:
                break

    return samples


# =============================================================================
# Moteurs
# =============================================================================
def run_moonshine(samples, args):
    """Transcrit tous les échantillons avec Moonshine."""
    from moonshine_engine import MoonshineEngine

    engine = MoonshineEngine(
        model_id=args.moonshine_model or None,
        device=args.moonshine_device,
        backend=args.moonshine_backend,
    )
    if not engine.load():
        print("[ERREUR] Chargement de Moonshine impossible")
        return None

    results = []
    for i, (audio, ref) in enumerate(samples, 1):
        t0 = time.perf_counter()
        hyp = engine.transcribe(audio)
        results.append((hyp, time.perf_counter() - t0, len(audio) / SAMPLE_RATE))
        if i % 25 == 0:
            print(f"    {i}/{len(samples)}...")
    return results


def run_whisper(samples, args):
    """Transcrit tous les échantillons avec faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  [WARN] faster-whisper non installé — comparaison Whisper ignorée")
        return None

    model_path = args.whisper_model
    if not os.path.isdir(model_path):
        print(f"  [WARN] Modèle CT2 introuvable ({model_path}) — "
              "utilisation de 'large-v3'")
        model_path = "large-v3"

    model = WhisperModel(model_path, device=args.whisper_device,
                         compute_type=args.whisper_compute)

    # Même conditionnement que l'application
    prompt = None
    vocab_file = os.path.join(BASE_DIR, "vocabulaire.txt")
    if os.path.isfile(vocab_file):
        with open(vocab_file, encoding="utf-8") as f:
            prompt = " ".join(
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ) or None

    results = []
    for i, (audio, ref) in enumerate(samples, 1):
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language="fr", beam_size=5, vad_filter=False,
            initial_prompt=prompt,
        )
        hyp = " ".join(s.text.strip() for s in segments).strip()
        results.append((hyp, time.perf_counter() - t0, len(audio) / SAMPLE_RATE))
        if i % 25 == 0:
            print(f"    {i}/{len(samples)}...")
    return results


# =============================================================================
# Rapport
# =============================================================================
def build_directory(samples) -> list[str]:
    """Construit l'annuaire des noms propres à partir des références.

    En production, la liste des médecins est connue : c'est ce que simule cet
    annuaire, pour mesurer ce que le fuzzy matching peut rattraper en aval.
    """
    names = set()
    for _, ref in samples:
        for tok in re.findall(r"\b[A-ZÀ-Ý][A-ZÀ-Ý'\-]{2,}\b", ref):
            names.add(tok)
    return sorted(names)


def apply_fuzzy(results, directory, threshold):
    """Applique la correction fuzzy aux hypothèses, comme le fait l'app."""
    from fuzzy_correction import fuzzy_match_names

    return [
        (fuzzy_match_names(hyp, directory, threshold), elapsed, duration)
        for hyp, elapsed, duration in results
    ]


def score(samples, results):
    """Calcule WER moyen, précision sur les noms propres et latences."""
    wers, latencies, rtfs = [], [], []
    nouns_total = nouns_hit = 0

    for (audio, ref), (hyp, elapsed, duration) in zip(samples, results):
        wers.append(word_error_rate(ref, hyp))
        latencies.append(elapsed)
        if duration > 0:
            rtfs.append(elapsed / duration)

        hyp_norm = normalize(hyp)
        for noun in extract_proper_nouns(ref):
            nouns_total += 1
            if noun in hyp_norm.split():
                nouns_hit += 1

    return {
        "wer": 100 * float(np.mean(wers)),
        "latency_ms": 1000 * float(np.median(latencies)),
        "rtf": float(np.mean(rtfs)) if rtfs else float("nan"),
        "nouns_total": nouns_total,
        "nouns_hit": nouns_hit,
        "nouns_pct": 100 * nouns_hit / nouns_total if nouns_total else float("nan"),
    }


def print_report(name, s):
    print(f"\n  {name}")
    print(f"    WER moyen          : {s['wer']:.2f}%")
    print(f"    Noms propres       : {s['nouns_hit']}/{s['nouns_total']} "
          f"({s['nouns_pct']:.1f}%)")
    print(f"    Latence médiane    : {s['latency_ms']:.0f} ms")
    print(f"    RTF moyen          : {s['rtf']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Banc d'essai Moonshine vs Whisper")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--limit", type=int, default=100,
                        help="Nombre d'échantillons (0 = tous, défaut: 100)")
    parser.add_argument("--moonshine_model", default="",
                        help="Modèle Moonshine (défaut: Cornebidouil/moonshine-tiny-fr)")
    parser.add_argument("--moonshine_device", default="auto")
    parser.add_argument("--moonshine_backend", default="torch", choices=["torch", "onnx"])
    parser.add_argument("--whisper_model", default=DEFAULT_CT2)
    parser.add_argument("--whisper_device", default="cuda")
    parser.add_argument("--whisper_compute", default="float16")
    parser.add_argument("--no_whisper", action="store_true")
    parser.add_argument("--show", type=int, default=10,
                        help="Nombre d'exemples détaillés à afficher (défaut: 10)")
    parser.add_argument("--fuzzy", action="store_true",
                        help="Mesurer aussi les scores APRÈS correction fuzzy, "
                             "avec un annuaire reconstruit depuis les références")
    parser.add_argument("--fuzzy_threshold", type=int, default=65,
                        help="Seuil du fuzzy matching (défaut: 65, comme config.json)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Banc d'essai Moonshine vs Whisper")
    print("=" * 64 + "\n")

    samples = load_samples(args)
    if not samples:
        print("\n[ERREUR] Aucun échantillon chargé")
        sys.exit(1)

    total_audio = sum(len(a) for a, _ in samples) / SAMPLE_RATE
    print(f"  {len(samples)} échantillons — {total_audio / 60:.1f} min d'audio\n")

    print("  Moonshine...")
    ms_results = run_moonshine(samples, args)
    if ms_results is None:
        sys.exit(1)

    wh_results = None
    if not args.no_whisper:
        print("\n  Whisper (faster-whisper)...")
        wh_results = run_whisper(samples, args)

    print("\n" + "=" * 64)
    print("  RÉSULTATS")
    print("=" * 64)

    ms_score = score(samples, ms_results)
    print_report("Moonshine", ms_score)

    if wh_results:
        wh_score = score(samples, wh_results)
        print_report("Whisper", wh_score)

        print("\n  Écarts (Moonshine - Whisper) :")
        print(f"    WER            : {ms_score['wer'] - wh_score['wer']:+.2f} points")
        print(f"    Noms propres   : {ms_score['nouns_pct'] - wh_score['nouns_pct']:+.1f} points")
        speedup = wh_score["latency_ms"] / ms_score["latency_ms"] if ms_score["latency_ms"] else float("nan")
        print(f"    Vitesse        : x{speedup:.1f}")

    # --- Après correction fuzzy ---
    if args.fuzzy:
        directory = build_directory(samples)
        print("\n" + "=" * 64)
        print(f"  APRÈS FUZZY (annuaire de {len(directory)} noms, seuil {args.fuzzy_threshold})")
        print("=" * 64)

        ms_fuzzy = apply_fuzzy(ms_results, directory, args.fuzzy_threshold)
        ms_fscore = score(samples, ms_fuzzy)
        print_report("Moonshine + fuzzy", ms_fscore)
        print(f"    -> gain noms propres : {ms_fscore['nouns_pct'] - ms_score['nouns_pct']:+.1f} points")

        if wh_results:
            wh_fuzzy = apply_fuzzy(wh_results, directory, args.fuzzy_threshold)
            wh_fscore = score(samples, wh_fuzzy)
            print_report("Whisper + fuzzy", wh_fscore)
            print(f"    -> gain noms propres : {wh_fscore['nouns_pct'] - wh_score['nouns_pct']:+.1f} points")
            print(f"\n  Écart résiduel sur les noms propres : "
                  f"{ms_fscore['nouns_pct'] - wh_fscore['nouns_pct']:+.1f} points")

        ms_results_display = ms_fuzzy
    else:
        ms_results_display = ms_results

    # --- Exemples détaillés ---
    if args.show:
        print("\n" + "=" * 64)
        print(f"  EXEMPLES ({min(args.show, len(samples))} premiers)")
        print("=" * 64)
        for i in range(min(args.show, len(samples))):
            ref = samples[i][1]
            print(f"\n  [{i + 1}] réf       : {ref}")
            print(f"      moonshine : {ms_results[i][0]}")
            if args.fuzzy:
                print(f"      + fuzzy   : {ms_results_display[i][0]}")
            if wh_results:
                print(f"      whisper   : {wh_results[i][0]}")

    print()


if __name__ == "__main__":
    main()
