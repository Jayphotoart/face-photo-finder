import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        id,
        photo_id,
        face_index,
        octet_length(embedding) AS embedding_bytes
    FROM face_embeddings
    ORDER BY id;
""")

rows = cursor.fetchall()

print("\nFACE EMBEDDINGS:")
print("-" * 50)

if not rows:
    print("No embeddings found.")
else:
    for row in rows:
        print(
            f"ID: {row[0]} | "
            f"Photo ID: {row[1]} | "
            f"Face: {row[2]} | "
            f"Bytes: {row[3]}"
        )

cursor.close()
conn.close()