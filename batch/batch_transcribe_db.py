#!/usr/bin/env python3
"""
Transcrit les fichiers audio du dossier fine_tuning/data/audio/
et met à jour le champ complete_transcription dans calls3 (MariaDB).
"""

import os
import re
import json
import time
import pymysql
from faster_whisper import WhisperModel

# --- Configuration ---
AUDIO_DIR = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\audio"
CONFIG_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\config.json"
VOCAB_PATH = r"C:\Users\FrédéricJamoulle\Claude\whisper\vocabulaire.txt"

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Bidochon;1",
    "database": "vocabase",
}

# Regex pour extraire le nom du .wav depuis le champ audio HTML
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


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def get_audio_to_callid_map(conn):
    """Retourne un dict {nom_fichier_wav: call_id} depuis calls3."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT call_id, audio FROM calls3 "
        "WHERE complete_transcription IS NULL OR complete_transcription = ''"
    )
    mapping = {}
    for call_id, audio_html in cursor:
        match = WAV_REGEX.search(audio_html)
        if match:
            wav_name = match.group(1)
            mapping[wav_name] = call_id
    cursor.close()
    return mapping


def main():
    config = load_config()
    vocab = load_vocabulary()

    # Lister les fichiers audio disponibles
    audio_files = {f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")}
    print(f"Fichiers audio dans le dossier : {len(audio_files)}")

    # Charger le modèle Whisper
    model_path = config.get("model_size", "large-v3-turbo")
    device = config.get("device", "cuda")
    compute_type = config.get("compute_type", "float16")
    language = config.get("language", "fr")

    print(f"Chargement du modèle '{model_path}' sur {device} ({compute_type})...")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print("Modèle chargé.")

    # Récupérer le mapping wav -> call_id depuis la DB
    conn = get_db_connection()
    wav_to_callid = get_audio_to_callid_map(conn)
    print(f"Enregistrements DB sans transcription complète : {len(wav_to_callid)}")

    # Intersection : fichiers présents dans le dossier ET dans la DB
    to_process = audio_files & set(wav_to_callid.keys())
    print(f"Fichiers à traiter (présents dans dossier + DB) : {len(to_process)}")

    if not to_process:
        print("Rien à transcrire.")
        conn.close()
        return

    cursor = conn.cursor()
    done = 0
    errors = 0
    total = len(to_process)
    start_time = time.time()

    for wav_name in sorted(to_process):
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

            cursor.execute(
                "UPDATE calls3 SET complete_transcription = %s WHERE call_id = %s",
                (text, call_id),
            )
            conn.commit()
            done += 1

            elapsed = time.time() - start_time
            avg = elapsed / done
            remaining = avg * (total - done)
            print(
                f"[{done}/{total}] {wav_name} -> {len(text)} chars "
                f"(~{remaining/60:.0f} min restantes)"
            )

        except Exception as e:
            errors += 1
            print(f"[ERREUR] {wav_name} : {e}")

    cursor.close()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\nTerminé ! {done} transcriptions en {elapsed/60:.1f} min ({errors} erreurs)")


if __name__ == "__main__":
    main()
