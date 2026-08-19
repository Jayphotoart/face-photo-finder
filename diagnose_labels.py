import psycopg2
import numpy as np
from itertools import combinations


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# =========================
# Connect
# =========================

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

print()
print("================================")
print("LABEL DIAGNOSTIC")
print("================================")
print()


# =========================
# Load Data
# =========================

cursor.execute("""
    SELECT
        fe.id,
        p.filename,
        fe.face_index,
        fe.embedding,
        fl.person_label
    FROM face_embeddings fe
    JOIN photos p
        ON fe.photo_id = p.id
    JOIN face_labels fl
        ON fe.id = fl.embedding_id
    ORDER BY fe.id;
""")

rows = cursor.fetchall()

faces = []

for row in rows:

    embedding_id = row[0]
    filename = row[1]
    face_index = row[2]
    embedding_bytes = row[3]
    label = row[4]

    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    )

    if embedding.shape != (512,):
        continue

    faces.append({
        "id": embedding_id,
        "filename": filename,
        "face_index": face_index,
        "label": label,
        "embedding": embedding
    })


print(f"Labeled faces loaded: {len(faces)}")


# =========================
# Calculate Pairs
# =========================

same_pairs = []
different_pairs = []

for a, b in combinations(faces, 2):

    score = cosine_similarity(
        a["embedding"],
        b["embedding"]
    )

    pair = {
        "score": score,
        "a": a,
        "b": b
    }

    if a["label"] == b["label"]:
        same_pairs.append(pair)

    else:
        different_pairs.append(pair)


# =========================
# Highest DIFFERENT-PERSON
# =========================

different_pairs.sort(
    key=lambda x: x["score"],
    reverse=True
)


print()
print("================================")
print("TOP 20 DIFFERENT-PERSON PAIRS")
print("================================")

for i, pair in enumerate(
    different_pairs[:20],
    start=1
):

    a = pair["a"]
    b = pair["b"]

    print()
    print(
        f"#{i}  Similarity: "
        f"{pair['score']:.6f}"
    )

    print(
        f"  {a['label']} | "
        f"ID {a['id']} | "
        f"{a['filename']} | "
        f"Face {a['face_index'] + 1}"
    )

    print(
        f"  {b['label']} | "
        f"ID {b['id']} | "
        f"{b['filename']} | "
        f"Face {b['face_index'] + 1}"
    )


# =========================
# Lowest SAME-PERSON
# =========================

same_pairs.sort(
    key=lambda x: x["score"]
)


print()
print("================================")
print("20 LOWEST SAME-PERSON PAIRS")
print("================================")

for i, pair in enumerate(
    same_pairs[:20],
    start=1
):

    a = pair["a"]
    b = pair["b"]

    print()
    print(
        f"#{i}  Similarity: "
        f"{pair['score']:.6f}"
    )

    print(
        f"  {a['label']} | "
        f"ID {a['id']} | "
        f"{a['filename']} | "
        f"Face {a['face_index'] + 1}"
    )

    print(
        f"  {b['label']} | "
        f"ID {b['id']} | "
        f"{b['filename']} | "
        f"Face {b['face_index'] + 1}"
    )


# =========================
# Label Counts
# =========================

print()
print("================================")
print("LABEL COUNTS")
print("================================")

cursor.execute("""
    SELECT
        person_label,
        COUNT(*)
    FROM face_labels
    GROUP BY person_label
    ORDER BY person_label;
""")

for label, count in cursor.fetchall():

    print(
        f"Person {label}: {count}"
    )


cursor.close()
conn.close()

print()
print("================================")
print("DIAGNOSTIC COMPLETE")
print("================================")