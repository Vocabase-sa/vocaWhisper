import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='Bidochon;1', database='vocabase')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM calls3 WHERE complete_transcription IS NOT NULL AND complete_transcription != ''")
done = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM calls3")
total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM calls3 WHERE complete_transcription IS NULL OR complete_transcription = ''")
remaining = cur.fetchone()[0]

print(f"Done: {done}/{total} ({100*done/total:.1f}%)")
print(f"Remaining: {remaining}")

# Last 5 updated rows
cur.execute("""
    SELECT call_id, LEFT(complete_transcription, 60)
    FROM calls3
    WHERE complete_transcription IS NOT NULL AND complete_transcription != ''
    ORDER BY call_id DESC
    LIMIT 5
""")
print("\nLast 5 completed:")
for row in cur.fetchall():
    print(f"  {row[0]} -> {row[1]}...")

conn.close()
