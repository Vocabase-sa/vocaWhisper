#!/usr/bin/env python3
"""
Export CSV pour fine-tuning Vosk ASR.
Format: call_id|filename|transcription|site_code|duration|confidence_score|created_date
"""

import pymysql
import os
import wave
import re

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Bidochon;1",
    "database": "vocabase",
}
AUDIO_DIR = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\audio\monitor"
OUTPUT = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\hospital_data_export.csv"

WAV_REGEX = re.compile(r"monitor/([^'\"]+\.wav)")
CONFIDENCE_REGEX = re.compile(r"\((\d+)%\)")

MIN_DURATION = 3.0
MAX_DURATION = 70.0


def get_wav_duration(filepath):
    """Retourne la durée du WAV en secondes, ou None si erreur."""
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return None
            return frames / rate
    except:
        return None


def extract_wav_name(audio_html):
    """Extrait le nom du fichier WAV depuis le champ audio HTML."""
    if not audio_html:
        return None
    m = WAV_REGEX.search(audio_html)
    return m.group(1) if m else None


def clean_stt_transcription(stt):
    """Nettoie la transcription STT : retire les pourcentages et <br>."""
    if not stt:
        return ""
    # "la cardiologie (97%)<br>ces rendez-vous (73%)" -> "la cardiologie ces rendez-vous"
    text = re.sub(r"<br>", " ", stt)
    text = re.sub(r"\s*\(\d+%\)", "", text)
    return text.strip()


def extract_confidence(stt):
    """Extrait le score de confiance moyen depuis la transcription STT."""
    if not stt:
        return ""
    scores = [int(m) for m in CONFIDENCE_REGEX.findall(stt)]
    if not scores:
        return ""
    return f"{sum(scores) / len(scores):.0f}"


def extract_site_code(call_id):
    """Extrait le code site depuis le call_id (ex: HEALTH01_xxx -> HEALTH01)."""
    parts = call_id.split("_")
    return parts[0] if parts else ""


def main():
    # --- Index des fichiers audio existants ---
    print("Indexation des fichiers audio...")
    audio_files = set(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"  {len(audio_files)} fichiers WAV sur disque")

    # --- Requête DB ---
    print("Requête DB...")
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT call_id, date, audio, transcription, complete_transcription
        FROM calls3
        WHERE (transcription IS NOT NULL AND transcription != '')
           OR (complete_transcription IS NOT NULL AND complete_transcription != ''
               AND complete_transcription NOT LIKE '[%')
        ORDER BY call_id
    """)
    rows = cur.fetchall()
    print(f"  {len(rows)} enregistrements avec transcription")
    conn.close()

    # --- Génération du CSV ---
    print("Génération du CSV...")
    stats = {
        "total_rows": len(rows),
        "no_wav_name": 0,
        "no_file": 0,
        "no_duration": 0,
        "too_short": 0,
        "too_long": 0,
        "skip_markers": 0,
        "no_transcription": 0,
        "exported": 0,
        "used_whisper": 0,
        "used_stt": 0,
        "sites": {},
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        # Header
        f.write("call_id|filename|transcription|site_code|duration|confidence_score|created_date\n")

        for call_id, date, audio, stt, complete in rows:
            # 1. Extraire le nom du WAV
            wav_name = extract_wav_name(audio)
            if not wav_name:
                stats["no_wav_name"] += 1
                continue

            # 2. Vérifier que le fichier existe
            if wav_name not in audio_files:
                stats["no_file"] += 1
                continue

            # 3. Mesurer la durée
            filepath = os.path.join(AUDIO_DIR, wav_name)
            duration = get_wav_duration(filepath)
            if duration is None:
                stats["no_duration"] += 1
                continue

            # 4. Filtrer par durée
            if duration < MIN_DURATION:
                stats["too_short"] += 1
                continue
            if duration > MAX_DURATION:
                stats["too_long"] += 1
                continue

            # 5. Choisir la transcription (Whisper en priorité)
            transcription = ""
            if complete and complete.strip() and not complete.startswith("["):
                transcription = complete.strip()
                stats["used_whisper"] += 1
            elif stt and stt.strip():
                transcription = clean_stt_transcription(stt)
                stats["used_stt"] += 1
            else:
                stats["no_transcription"] += 1
                continue

            if not transcription:
                stats["no_transcription"] += 1
                continue

            # 6. Métadonnées
            site_code = extract_site_code(call_id)
            confidence = extract_confidence(stt) if stt else ""
            date_str = date.strftime("%Y-%m-%d %H:%M:%S") if date else ""

            # 7. Nettoyer la transcription (retirer les pipes et newlines)
            transcription = transcription.replace("|", " ").replace("\n", " ").replace("\r", " ")

            # 8. Écrire la ligne
            f.write(f"{call_id}|{wav_name}|{transcription}|{site_code}|{duration:.1f}|{confidence}|{date_str}\n")
            stats["exported"] += 1
            stats["sites"][site_code] = stats["sites"].get(site_code, 0) + 1

    # --- Statistiques ---
    print(f"\n{'='*60}")
    print(f"EXPORT TERMINE : {OUTPUT}")
    print(f"{'='*60}")
    print(f"  Lignes DB avec transcription : {stats['total_rows']:,}")
    print(f"  Pas de WAV name              : {stats['no_wav_name']:,}")
    print(f"  Fichier absent               : {stats['no_file']:,}")
    print(f"  Duree illisible              : {stats['no_duration']:,}")
    print(f"  Trop court (< {MIN_DURATION}s)        : {stats['too_short']:,}")
    print(f"  Trop long (> {MAX_DURATION}s)         : {stats['too_long']:,}")
    print(f"  Pas de transcription valide  : {stats['no_transcription']:,}")
    print(f"  ------------------------------")
    print(f"  EXPORTÉS                     : {stats['exported']:,}")
    print(f"    dont Whisper               : {stats['used_whisper']:,}")
    print(f"    dont STT nettoyé           : {stats['used_stt']:,}")
    print(f"\n  Repartition par site :")
    for site, cnt in sorted(stats["sites"].items()):
        print(f"    {site} : {cnt:,}")

    # Vérifier le fichier
    file_size = os.path.getsize(OUTPUT)
    print(f"\n  Taille fichier : {file_size / 1_000_000:.1f} MB")

    # Afficher les 3 premières lignes
    print(f"\n  Aperçu (3 premières lignes) :")
    with open(OUTPUT, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 4:
                break
            print(f"    {line.rstrip()}")


if __name__ == "__main__":
    main()
