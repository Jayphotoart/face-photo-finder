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

# Old duplicate test.jpg
PHOTO_ID_TO_DELETE = 1

# Check first
cursor.execute("""
    SELECT id, filename
    FROM photos
    WHERE id = %s;
""", (PHOTO_ID_TO_DELETE,))

photo = cursor.fetchone()

if photo is None:
    print("Photo ID 1 not found.")
    cursor.close()
    conn.close()
    exit()

print(f"Found: ID={photo[0]}, filename={photo[1]}")

if photo[1].lower() != "test.jpg":
    print("Safety check failed. Nothing deleted.")
    cursor.close()
    conn.close()
    exit()

# Delete associated embeddings first
cursor.execute("""
    DELETE FROM face_embeddings
    WHERE photo_id = %s;
""", (PHOTO_ID_TO_DELETE,))

deleted_embeddings = cursor.rowcount

# Delete photo
cursor.execute("""
    DELETE FROM photos
    WHERE id = %s
    AND LOWER(filename) = 'test.jpg';
""", (PHOTO_ID_TO_DELETE,))

deleted_photo = cursor.rowcount

conn.commit()

print()
print("================================")
print("CLEANUP COMPLETE")
print("================================")
print(f"Embeddings deleted: {deleted_embeddings}")
print(f"Photo deleted:      {deleted_photo}")
print("================================")

cursor.close()
conn.close()