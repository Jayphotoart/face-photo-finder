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
    cur.execute("""
        SELECT
            person_label,
            COUNT(*) AS total
        FROM face_labels
        GROUP BY person_label
        ORDER BY person_label;
    """)

    rows = cur.fetchall()

print("\nLABEL SUMMARY")
print("=" * 30)

total = 0

for label, count in rows:
    print(f"{label:6} : {count}")
    total += count

print("=" * 30)
print(f"TOTAL  : {total}")

conn.close()