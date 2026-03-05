#!/usr/bin/env python3
"""
Transcrit tous les fichiers audio du dossier fine_tuning/data/audio/
et écrit les résultats dans un CSV.
Reprend automatiquement là où il s'est arrêté si le CSV existe déjà.
"""

import csv
import json
import os
import time

from faster_whisper import WhisperModel

# --- Configuration ---
AUDIO_DIR = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\audio"
CONFIG_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\config.json"
VOCAB_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\vocabulaire.txt"
OUTPUT_CSV = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\transcriptions_complete.csv"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_vocabulary():
    if not os.path.exists(VOCAB_PATH):
        return ""
    with open(VOCAB_PATH, encoding="utf-8") as f:
        words = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return ", ".join(words)


def load_already_done(csv_path):
    """Charge les fichiers déjà transcrits depuis le CSV existant."""
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["audio_file"]] = row["transcription"]
    return done


def main():
    config = load_config()
    vocab = load_vocabulary()

    # Lister les fichiers audio
    audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"Fichiers audio dans le dossier : {len(audio_files)}")

    # Charger les transcriptions déjà faites (reprise)
    already_done = load_already_done(OUTPUT_CSV)
    if already_done:
        print(f"Déjà transcrits (reprise) : {len(already_done)}")

    to_process = [f for f in audio_files if f not in already_done]
    print(f"Restant à transcrire : {len(to_process)}")

    if not to_process:
        print("Tout est déjà transcrit.")
        return

    # Charger le modèle Whisper
    model_path = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    compute_type = config.get("compute_type", "float16")
    language = config.get("language", "fr")

    print(f"Chargement du modèle '{model_path}' sur {device} ({compute_type})...")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print("Modèle chargé.")

    # Ouvrir le CSV en mode append
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    csvfile = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["audio_file", "transcription"])

    done = 0
    errors = 0
    total = len(to_process)
    start_time = time.time()

    for wav_name in to_process:
        filepath = os.path.join(AUDIO_DIR, wav_name)

        try:
            segments, info = model.transcribe(
                filepath,
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=vocab,
            )
            text = " ".join(s.text.strip() for s in segments).strip()

            writer.writerow([wav_name, text])
            csvfile.flush()
            done += 1

            elapsed = time.time() - start_time
            avg = elapsed / done
            remaining = avg * (total - done)
            print(
                f"[{done}/{total}] {wav_name} -> {len(text)} chars "
                f"(~{remaining / 60:.0f} min restantes)"
            )

        except Exception as e:
            errors += 1
            print(f"[ERREUR] {wav_name} : {e}")

    csvfile.close()

    elapsed = time.time() - start_time
    print(
        f"\nTerminé ! {done} transcriptions en {elapsed / 60:.1f} min ({errors} erreurs)"
    )
    print(f"Résultat : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
