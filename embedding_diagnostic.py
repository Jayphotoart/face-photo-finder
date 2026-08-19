
import numpy as np
import psycopg2


# ============================================================
# CONFIG
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

TARGET_FILE = "SAI_5929.jpg"


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return None

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


# ============================================================
# LOAD DATABASE
# ============================================================

print("=" * 70)
print("EMBEDDING INTEGRITY DIAGNOSTIC")
print("=" * 70)

print()
print("Connecting to PostgreSQL...")

try:

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)

    raise SystemExit


# ============================================================
# LOAD ALL LABELED EMBEDDINGS
# ============================================================

cursor.execute("""
    SELECT
        fe.id,
        fe.photo_id,
        fe.face_index,
        fe.embedding,
        p.filename,
        fl.person_label
    FROM face_embeddings fe

    JOIN photos p
        ON p.id = fe.photo_id

    JOIN face_labels fl
        ON fl.embedding_id = fe.id

    WHERE fl.person_label IN ('A', 'B', 'C', 'D')

    ORDER BY fe.id;
""")

rows = cursor.fetchall()


print()
print(
    "Labeled embeddings:",
    len(rows)
)


# ============================================================
# PREPARE DATA
# ============================================================

faces = []


for row in rows:

    embedding_id = row[0]
    photo_id = row[1]
    face_index = row[2]
    embedding_data = row[3]
    filename = row[4]
    person_label = row[5]


    embedding = np.frombuffer(
        embedding_data,
        dtype=np.float32
    ).copy()


    norm = np.linalg.norm(
        embedding
    )


    faces.append({

        "id": embedding_id,

        "photo_id": photo_id,

        "face_index": face_index,

        "filename": filename,

        "person": person_label,

        "embedding": embedding,

        "dimension": len(embedding),

        "norm": float(norm)
    })


# ============================================================
# BASIC EMBEDDING INTEGRITY
# ============================================================

print()
print("=" * 70)
print("EMBEDDING BASIC INTEGRITY")
print("=" * 70)


for face in faces:

    status = "OK"


    if face["dimension"] != 512:

        status = "BAD_DIMENSION"


    if face["norm"] == 0:

        status = "ZERO_VECTOR"


    print(
        f"ID={face['id']:>3} | "
        f"{face['filename']:<30} | "
        f"Face={face['face_index']} | "
        f"Person={face['person']} | "
        f"Dim={face['dimension']} | "
        f"Norm={face['norm']:.6f} | "
        f"{status}"
    )


# ============================================================
# TARGET FILE
# ============================================================

target_faces = [

    face
    for face in faces
    if face["filename"].lower()
    == TARGET_FILE.lower()
]


print()
print("=" * 70)
print(f"TARGET FILE: {TARGET_FILE}")
print("=" * 70)


if not target_faces:

    print()
    print(
        "ERROR: Target file was not found."
    )

    cursor.close()
    conn.close()

    raise SystemExit


print(
    f"Faces found in database: "
    f"{len(target_faces)}"
)


for face in target_faces:

    print(
        f"Face ID={face['id']} | "
        f"Face Index={face['face_index']} | "
        f"Person={face['person']} | "
        f"Dimension={face['dimension']} | "
        f"Norm={face['norm']:.6f}"
    )


# ============================================================
# TARGET FACE vs SAME PERSON
# ============================================================

print()
print("=" * 70)
print("SAI_5929 FACE → SAME PERSON COMPARISON")
print("=" * 70)


for target in target_faces:

    same_person = [

        face
        for face in faces

        if face["person"] == target["person"]

        and face["id"] != target["id"]

        and face["filename"].lower()
        != TARGET_FILE.lower()
    ]


    similarities = []


    for other in same_person:

        sim = cosine_similarity(
            target["embedding"],
            other["embedding"]
        )


        if sim is not None:

            similarities.append({

                "similarity": sim,

                "filename": other["filename"],

                "face_index": other["face_index"],

                "id": other["id"]
            })


    similarities.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    print()
    print(
        f"TARGET: "
        f"Face={target['face_index']} | "
        f"Person={target['person']} | "
        f"ID={target['id']}"
    )


    if not similarities:

        print("No same-person reference faces.")

        continue


    for item in similarities:

        print(
            f"   Similarity={item['similarity']:.6f} | "
            f"{item['filename']} | "
            f"Face={item['face_index']} | "
            f"ID={item['id']}"
        )


    print(
        f"   BEST SAME-PERSON = "
        f"{similarities[0]['similarity']:.6f}"
    )

    print(
        f"   WORST SAME-PERSON = "
        f"{similarities[-1]['similarity']:.6f}"
    )


# ============================================================
# TARGET FACE vs ALL OTHER PEOPLE
# ============================================================

print()
print("=" * 70)
print("SAI_5929 FACE → DIFFERENT PERSON COMPARISON")
print("=" * 70)


for target in target_faces:

    different_person = [

        face
        for face in faces

        if face["person"] != target["person"]

        and face["id"] != target["id"]
    ]


    similarities = []


    for other in different_person:

        sim = cosine_similarity(
            target["embedding"],
            other["embedding"]
        )


        if sim is not None:

            similarities.append({

                "similarity": sim,

                "filename": other["filename"],

                "face_index": other["face_index"],

                "person": other["person"],

                "id": other["id"]
            })


    similarities.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    print()
    print(
        f"TARGET: "
        f"Face={target['face_index']} | "
        f"Person={target['person']} | "
        f"ID={target['id']}"
    )


    for item in similarities[:5]:

        print(
            f"   Similarity={item['similarity']:.6f} | "
            f"Person={item['person']} | "
            f"{item['filename']} | "
            f"Face={item['face_index']} | "
            f"ID={item['id']}"
        )


    if similarities:

        print(
            f"   HIGHEST DIFFERENT-PERSON = "
            f"{similarities[0]['similarity']:.6f}"
        )


# ============================================================
# SAI_5929 INTERNAL FACE-TO-FACE COMPARISON
# ============================================================

print()
print("=" * 70)
print("SAI_5929 INTERNAL FACE-TO-FACE SIMILARITY")
print("=" * 70)


for i in range(len(target_faces)):

    for j in range(i + 1, len(target_faces)):

        a = target_faces[i]
        b = target_faces[j]


        sim = cosine_similarity(
            a["embedding"],
            b["embedding"]
        )


        print(
            f"Face {a['face_index']} "
            f"(Person {a['person']})"
            f"  ↔  "
            f"Face {b['face_index']} "
            f"(Person {b['person']})"
            f"  =  "
            f"{sim:.6f}"
        )


# ============================================================
# DUPLICATE / IDENTICAL EMBEDDING CHECK
# ============================================================

print()
print("=" * 70)
print("IDENTICAL / NEAR-IDENTICAL EMBEDDING CHECK")
print("=" * 70)


near_duplicates = []


for i in range(len(faces)):

    for j in range(i + 1, len(faces)):

        a = faces[i]
        b = faces[j]


        sim = cosine_similarity(
            a["embedding"],
            b["embedding"]
        )


        if sim is not None and sim >= 0.9999:

            near_duplicates.append({

                "similarity": sim,

                "a": a,

                "b": b
            })


if not near_duplicates:

    print(
        "No near-identical embeddings found."
    )

else:

    for item in near_duplicates:

        a = item["a"]
        b = item["b"]


        print(
            f"Similarity={item['similarity']:.6f}"
        )

        print(
            f"   {a['filename']} | "
            f"Face={a['face_index']} | "
            f"Person={a['person']} | "
            f"ID={a['id']}"
        )

        print(
            f"   {b['filename']} | "
            f"Face={b['face_index']} | "
            f"Person={b['person']} | "
            f"ID={b['id']}"
        )


# ============================================================
# CLOSE
# ============================================================

cursor.close()
conn.close()


print()
print("=" * 70)
print("PostgreSQL connection closed.")
print("Diagnostic completed.")
print("=" * 70)

