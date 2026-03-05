#!/usr/bin/env python3
"""
Transcrit les fichiers audio de fine_tuning/data/audio/monitor/
et met à jour complete_transcription dans calls3 (MariaDB).
Reprend automatiquement là où il s'est arrêté.
Vérifie chaque fichier audio AVANT de le transcrire pour éviter les blocages.
"""

import json
import os
import re
import time
import wave
import subprocess
import sys

import pymysql
from faster_whisper import WhisperModel

# --- Configuration ---
AUDIO_DIR = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\audio\monitor"
CONFIG_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\config.json"
VOCAB_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\vocabulaire.txt"

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Bidochon;1",
    "database": "vocabase",
}

BATCH_COMMIT = 10
MIN_FILE_SIZE = 100       # ignorer fichiers trop petits
MAX_FILE_SIZE = 50_000_000  # 50 MB max
MAX_DURATION = 600        # 10 min max
MIN_DURATION = 0.1        # 100ms min

WAV_REGEX = re.compile(r"monitor/([^'\"]+\.wav)")


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


def check_wav_file(filepath):
    """Vérifie un fichier WAV. Retourne (ok, duration, error_msg)."""
    try:
        fsize = os.path.getsize(filepath)
        if fsize < MIN_FILE_SIZE:
            return False, 0, f"trop petit ({fsize} octets)"
        if fsize > MAX_FILE_SIZE:
            return False, 0, f"trop gros ({fsize / 1_000_000:.1f} MB)"

        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return False, 0, "framerate = 0"
            duration = frames / rate
            if duration < MIN_DURATION:
                return False, 0, f"trop court ({duration:.2f}s)"
            if duration > MAX_DURATION:
                return False, 0, f"trop long ({duration:.0f}s)"
            return True, duration, None
    except Exception as e:
        return False, 0, f"corrompu: {e}"


def build_mapping(conn):
    """Retourne un dict {wav_filename: call_id} pour les lignes sans transcription."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT call_id, audio FROM calls3 "
        "WHERE complete_transcription IS NULL OR complete_transcription = ''"
    )
    mapping = {}
    for call_id, audio_html in cursor:
        if audio_html:
            m = WAV_REGEX.search(audio_html)
            if m:
                mapping[m.group(1)] = call_id
    cursor.close()
    return mapping


def main():
    config = load_config()
    vocab = load_vocabulary()

    # Fichiers audio disponibles
    audio_files = set(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"Fichiers audio dans monitor : {len(audio_files)}")

    # Mapping DB
    conn = pymysql.connect(**DB_CONFIG)
    wav_to_callid = build_mapping(conn)
    print(f"Enregistrements DB sans transcription : {len(wav_to_callid)}")

    # Intersection
    to_process = sorted(audio_files & set(wav_to_callid.keys()))
    print(f"Fichiers à transcrire : {len(to_process)}")

    if not to_process:
        print("Rien à transcrire.")
        conn.close()
        return

    # Phase 1 : pré-vérification de tous les fichiers
    print("\n--- Pré-vérification des fichiers audio ---")
    valid_files = []
    skipped = 0
    for wav_name in to_process:
        filepath = os.path.join(AUDIO_DIR, wav_name)
        ok, duration, err = check_wav_file(filepath)
        if ok:
            valid_files.append((wav_name, duration))
        else:
            skipped += 1
            # Marquer comme [SKIP] pour ne pas retenter
            call_id = wav_to_callid[wav_name]
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE calls3 SET complete_transcription = %s WHERE call_id = %s",
                (f"[SKIP: {err}]", call_id),
            )
            cursor.close()
            if skipped % 100 == 0:
                conn.commit()
                print(f"  {skipped} fichiers skippés...")

    conn.commit()
    print(f"Pré-vérification terminée : {len(valid_files)} valides, {skipped} skippés")

    if not valid_files:
        print("Aucun fichier valide à transcrire.")
        conn.close()
        return

    # Phase 2 : transcription
    model_path = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    compute_type = config.get("compute_type", "float16")
    language = config.get("language", "fr")

    print(f"\nChargement du modèle '{model_path}' sur {device} ({compute_type})...")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print("Modèle chargé. Début de la transcription...\n")

    cursor = conn.cursor()
    done = 0
    errors = 0
    total = len(valid_files)
    start_time = time.time()

    for wav_name, duration in valid_files:
        filepath = os.path.join(AUDIO_DIR, wav_name)
        call_id = wav_to_callid[wav_name]

        try:
            segments, info = model.transcribe(
                filepath,
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=vocab,
            )
            text = " ".join(s.text.strip() for s in segments).strip()

            # Tronquer à 4096 chars
            if len(text) > 4096:
                text = text[:4096]

            cursor.execute(
                "UPDATE calls3 SET complete_transcription = %s WHERE call_id = %s",
                (text, call_id),
            )
            done += 1

            if done % BATCH_COMMIT == 0:
                conn.commit()

            elapsed = time.time() - start_time
            avg = elapsed / done
            remaining = avg * (total - done)
            print(
                f"[{done}/{total}] {wav_name} ({duration:.1f}s) -> {len(text)} chars "
                f"(~{remaining / 60:.0f} min rest. | {errors} err)",
                flush=True,
            )

        except Exception as e:
            errors += 1
            cursor.execute(
                "UPDATE calls3 SET complete_transcription = %s WHERE call_id = %s",
                (f"[ERROR: {e}]", call_id),
            )
            print(f"[ERREUR] {wav_name} : {e}", flush=True)

            if done % BATCH_COMMIT == 0:
                conn.commit()

    conn.commit()
    cursor.close()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\nTerminé ! {done} traités en {elapsed / 60:.1f} min")
    print(f"  - {done - errors} OK")
    print(f"  - {errors} erreurs")
    print(f"  - {skipped} skippés (pré-vérification)")


if __name__ == "__main__":
    main()
