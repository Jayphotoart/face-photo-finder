import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

conn = psycopg2.connect(**DB_CONFIG)

with conn.cursor() as cur:
    cur.execute("DELETE FROM face_labels;")
    cur.execute("SELECT COUNT(*) FROM face_labels;")
    count = cur.fetchone()[0]

conn.commit()
conn.close()

print(f"face_labels cleared. Remaining labels: {count}")