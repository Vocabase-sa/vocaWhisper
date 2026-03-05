import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='Bidochon;1', database='vocabase')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM calls3 WHERE complete_transcription IS NOT NULL AND complete_transcription != ''")
done = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM calls3")
total = cur.fetchone()[0]

print(f"Transcription progress: {done}/{total} ({100*done/total:.1f}%)")
conn.close()
