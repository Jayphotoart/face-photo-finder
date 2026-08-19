import psycopg2
import numpy as np
from itertools import combinations


# =========================
# PostgreSQL Configuration
# =========================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


# =========================
# Cosine Similarity
# =========================

def cosine_similarity(a, b):

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b) / (a_norm * b_norm)
    )


# =========================
# Database Connection
# =========================

print()
print("================================")
print("THRESHOLD CALIBRATION")
print("================================")
print()

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

print("PostgreSQL connected.")


# =========================
# Load Labeled Embeddings
# =========================

cursor.execute("""
    SELECT
        fe.id,
        fe.embedding,
        fl.person_label
    FROM face_embeddings fe
    JOIN face_labels fl
        ON fe.id = fl.embedding_id
    WHERE fl.person_label IN ('A', 'B', 'C', 'D')
    ORDER BY fe.id;
""")

rows = cursor.fetchall()

print(f"Labeled embeddings found: {len(rows)}")


if len(rows) < 2:

    print("Not enough labeled embeddings.")

    cursor.close()
    conn.close()

    exit()


# =========================
# Decode Embeddings
# =========================

faces = []

for embedding_id, embedding_bytes, person_label in rows:

    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    )

    if embedding.shape[0] != 512:

        print(
            f"WARNING: Embedding {embedding_id} "
            f"has dimension {embedding.shape[0]}"
        )

        continue

    faces.append({
        "id": embedding_id,
        "label": person_label,
        "embedding": embedding
    })


print(f"Valid 512D embeddings: {len(faces)}")


# =========================
# Calculate Pair Similarities
# =========================

same_person = []
different_person = []

for a, b in combinations(faces, 2):

    score = cosine_similarity(
        a["embedding"],
        b["embedding"]
    )

    if a["label"] == b["label"]:

        same_person.append(score)

    else:

        different_person.append(score)


# =========================
# Statistics
# =========================

print()
print("================================")
print("SIMILARITY STATISTICS")
print("================================")

print()

print(f"Same-person pairs:      {len(same_person)}")
print(f"Different-person pairs: {len(different_person)}")

print()

print("SAME PERSON")
print("--------------------------------")

if same_person:

    print(f"Minimum : {min(same_person):.4f}")
    print(f"Average : {np.mean(same_person):.4f}")
    print(f"Maximum : {max(same_person):.4f}")

print()

print("DIFFERENT PERSON")
print("--------------------------------")

if different_person:

    print(f"Minimum : {min(different_person):.4f}")
    print(f"Average : {np.mean(different_person):.4f}")
    print(f"Maximum : {max(different_person):.4f}")


# =========================
# Find Best Threshold
# =========================

print()
print("================================")
print("THRESHOLD TEST")
print("================================")
print()

best_threshold = None
best_accuracy = -1

all_pairs = []

for score in same_person:

    all_pairs.append(
        (score, 1)
    )

for score in different_person:

    all_pairs.append(
        (score, 0)
    )


for threshold in np.arange(
    0.30,
    0.91,
    0.01
):

    correct = 0

    for score, actual_same in all_pairs:

        predicted_same = (
            score >= threshold
        )

        if predicted_same == bool(actual_same):

            correct += 1

    accuracy = (
        correct / len(all_pairs)
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Pair accuracy   : "
    f"{best_accuracy * 100:.2f}%"
)


# =========================
# Test Best Threshold
# =========================

threshold = best_threshold

false_same = 0
false_different = 0

for score in same_person:

    if score < threshold:

        false_different += 1


for score in different_person:

    if score >= threshold:

        false_same += 1


print()
print("================================")
print("THRESHOLD RESULTS")
print("================================")

print(
    f"Threshold: {threshold:.2f}"
)

print(
    f"Same person incorrectly rejected: "
    f"{false_different}"
)

print(
    f"Different person incorrectly accepted: "
    f"{false_same}"
)


# =========================
# Close
# =========================

cursor.close()
conn.close()

print()
print("================================")
print("CALIBRATION COMPLETE")
print("================================")