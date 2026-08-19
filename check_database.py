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


# =========================
# 1. Photos Count
# =========================

cursor.execute("""
    SELECT COUNT(*)
    FROM photos;
""")

photo_count = cursor.fetchone()[0]

print()
print("================================")
print(f"TOTAL PHOTOS: {photo_count}")
print("================================")


# =========================
# 2. Face Embeddings Count
# =========================

cursor.execute("""
    SELECT COUNT(*)
    FROM face_embeddings;
""")

embedding_count = cursor.fetchone()[0]

print()
print("================================")
print(f"TOTAL FACE EMBEDDINGS: {embedding_count}")
print("================================")


# =========================
# 3. Faces Per Photo
# =========================

cursor.execute("""
    SELECT
        p.id,
        p.filename,
        COUNT(fe.id) AS face_count
    FROM photos p
    LEFT JOIN face_embeddings fe
        ON p.id = fe.photo_id
    GROUP BY
        p.id,
        p.filename
    ORDER BY p.id;
""")

rows = cursor.fetchall()

print()
print("================================")
print("FACES PER PHOTO")
print("================================")

for photo_id, filename, face_count in rows:

    print(
        f"ID={photo_id:<4} "
        f"{filename:<25} "
        f"Faces={face_count}"
    )


# =========================
# 4. Duplicate Filenames
# =========================

cursor.execute("""
    SELECT
        filename,
        COUNT(*) AS count
    FROM photos
    GROUP BY filename
    HAVING COUNT(*) > 1
    ORDER BY filename;
""")

duplicates = cursor.fetchall()

print()
print("================================")
print("DUPLICATE FILENAMES")
print("================================")

if duplicates:

    for filename, count in duplicates:
        print(
            f"{filename} -> {count} records"
        )

else:

    print("No duplicate filenames found.")


# =========================
# 5. test.jpg Records
# =========================

cursor.execute("""
    SELECT
        p.id,
        p.filename,
        COUNT(fe.id) AS face_count
    FROM photos p
    LEFT JOIN face_embeddings fe
        ON p.id = fe.photo_id
    WHERE LOWER(p.filename) = 'test.jpg'
    GROUP BY p.id, p.filename
    ORDER BY p.id;
""")

test_records = cursor.fetchall()

print()
print("================================")
print("TEST.JPG RECORDS")
print("================================")

if test_records:

    for photo_id, filename, face_count in test_records:

        print(
            f"ID={photo_id} | "
            f"{filename} | "
            f"Faces={face_count}"
        )

else:

    print("No test.jpg record found.")


# =========================
# 6. JAY02027.JPG Records
# =========================

cursor.execute("""
    SELECT
        p.id,
        p.filename,
        COUNT(fe.id) AS face_count
    FROM photos p
    LEFT JOIN face_embeddings fe
        ON p.id = fe.photo_id
    WHERE LOWER(p.filename) = 'jay02027.jpg'
    GROUP BY p.id, p.filename
    ORDER BY p.id;
""")

jay_records = cursor.fetchall()

print()
print("================================")
print("JAY02027.JPG RECORDS")
print("================================")

if jay_records:

    for photo_id, filename, face_count in jay_records:

        print(
            f"ID={photo_id} | "
            f"{filename} | "
            f"Faces={face_count}"
        )

else:

    print("No JAY02027.JPG record found.")


cursor.close()
conn.close()

print()
print("Database connection closed.")