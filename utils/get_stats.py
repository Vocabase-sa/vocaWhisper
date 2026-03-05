import pymysql
import os
import wave

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Bidochon;1",
    "database": "vocabase",
}
AUDIO_DIR = r"C:\Users\FrédéricJamoulle\Claude\whisper\fine_tuning\data\audio\monitor"

conn = pymysql.connect(**DB_CONFIG)
cur = conn.cursor()

# 1. Nombre total
cur.execute("SELECT COUNT(*) FROM calls3")
total = cur.fetchone()[0]
print(f"1. Total enregistrements DB : {total}")

# 2. Fichiers audio sur disque
wav_count = len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
print(f"   Fichiers WAV sur disque  : {wav_count}")

# 3. Transcriptions existantes
cur.execute("SELECT COUNT(*) FROM calls3 WHERE transcription IS NOT NULL AND transcription != ''")
stt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM calls3 WHERE complete_transcription IS NOT NULL AND complete_transcription != '' AND complete_transcription NOT LIKE '[%'")
whisper = cur.fetchone()[0]
print(f"   Transcriptions STT       : {stt}")
print(f"   Transcriptions Whisper   : {whisper}")

# 4. Durée moyenne (échantillon de 500 fichiers)
import random
all_wavs = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
sample = random.sample(all_wavs, min(500, len(all_wavs)))
durations = []
for f in sample:
    try:
        with wave.open(os.path.join(AUDIO_DIR, f), 'rb') as wf:
            dur = wf.getnframes() / wf.getframerate()
            if dur > 0.1:
                durations.append(dur)
    except:
        pass

if durations:
    avg = sum(durations) / len(durations)
    durations.sort()
    median = durations[len(durations)//2]
    print(f"\n4. Durées (échantillon {len(durations)} fichiers) :")
    print(f"   Moyenne  : {avg:.1f}s")
    print(f"   Médiane  : {median:.1f}s")
    print(f"   Min      : {min(durations):.1f}s")
    print(f"   Max      : {max(durations):.1f}s")
    print(f"   < 5s     : {sum(1 for d in durations if d < 5)}")
    print(f"   5-30s    : {sum(1 for d in durations if 5 <= d < 30)}")
    print(f"   30-60s   : {sum(1 for d in durations if 30 <= d < 60)}")
    print(f"   60-120s  : {sum(1 for d in durations if 60 <= d < 120)}")
    print(f"   > 120s   : {sum(1 for d in durations if d >= 120)}")

# 5. Speakers / sites
cur.execute("""
    SELECT SUBSTRING_INDEX(call_id, '_', 1) AS site, COUNT(*) AS cnt
    FROM calls3
    GROUP BY site
    ORDER BY cnt DESC
""")
print(f"\n5. Sites/speakers :")
for row in cur.fetchall():
    print(f"   {row[0]} : {row[1]} appels")

conn.close()
