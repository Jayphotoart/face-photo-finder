import os
import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


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
# Query Image
# =========================

QUERY_IMAGE = "images/test2.jpg"


# =========================
# Cosine Similarity
# =========================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

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
# Load Face Model
# =========================

print("Loading face model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Face model loaded.")


# =========================
# Read Query Image
# =========================

if not os.path.exists(QUERY_IMAGE):

    print(f"Query image not found: {QUERY_IMAGE}")
    exit()


image = cv2.imread(QUERY_IMAGE)

if image is None:

    print("Could not read query image.")
    exit()


print(f"Query image: {QUERY_IMAGE}")


# =========================
# Detect Query Faces
# =========================

faces = app.get(image)

print(f"Faces detected: {len(faces)}")


if len(faces) == 0:

    print("No face found in query image.")
    exit()


# For now, use the first detected face
query_embedding = np.asarray(
    faces[0].embedding,
    dtype=np.float32
)

print(
    f"Query embedding shape: "
    f"{query_embedding.shape}"
)


# =========================
# Database Connection
# =========================

try:

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("PostgreSQL connection successful!")

except Exception as e:

    print("Database connection failed:")
    print(e)
    exit()


# =========================
# Load Stored Embeddings
# =========================

cursor.execute("""
    SELECT
        fe.id,
        fe.photo_id,
        fe.face_index,
        fe.embedding,
        p.filename,
        p.storage_path
    FROM face_embeddings fe
    JOIN photos p
        ON fe.photo_id = p.id
    ORDER BY fe.id;
""")

rows = cursor.fetchall()

print(f"Stored face embeddings: {len(rows)}")


# =========================
# Compare
# =========================

results = []


for row in rows:

    embedding_id = row[0]
    photo_id = row[1]
    face_index = row[2]
    embedding_bytes = row[3]
    filename = row[4]
    storage_path = row[5]

    # Convert BYTEA → NumPy array
    stored_embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    )

    # Safety check
    if stored_embedding.shape != (512,):

        print(
            f"Skipping embedding ID {embedding_id}: "
            f"shape={stored_embedding.shape}"
        )

        continue


    # Calculate similarity
    score = cosine_similarity(
        query_embedding,
        stored_embedding
    )


    results.append({
        "embedding_id": embedding_id,
        "photo_id": photo_id,
        "face_index": face_index,
        "filename": filename,
        "storage_path": storage_path,
        "score": score
    })


# =========================
# Ranking
# =========================

results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# =========================
# Display Top Results
# =========================

print()
print("==============================================")
print("FACE SEARCH RESULTS")
print("==============================================")

print(
    f"{'Rank':<6}"
    f"{'Photo':<25}"
    f"{'Face':<8}"
    f"{'Similarity':<12}"
)

print("----------------------------------------------")


for rank, result in enumerate(
    results[:20],
    start=1
):

    print(
        f"{rank:<6}"
        f"{result['filename']:<25}"
        f"{result['face_index'] + 1:<8}"
        f"{result['score']:.6f}"
    )


print("==============================================")


# =========================
# Close Database
# =========================

cursor.close()
conn.close()

print("Database connection closed.")