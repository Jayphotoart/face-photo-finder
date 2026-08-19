import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


# =========================
# Load Model
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
# Get test.jpg embedding
# =========================

image = cv2.imread("images/test.jpg")

if image is None:
    print("Could not read images/test.jpg")
    exit()

faces = app.get(image)

if len(faces) == 0:

    print("No face found in test.jpg")
    exit()

query_embedding = np.asarray(
    faces[0].embedding,
    dtype=np.float32
)


# =========================
# Database
# =========================

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()


# =========================
# Get embeddings
# =========================

cursor.execute("""
    SELECT
        fe.id,
        p.id,
        p.filename,
        fe.embedding
    FROM face_embeddings fe
    JOIN photos p
        ON fe.photo_id = p.id
    WHERE LOWER(p.filename)
        IN ('test.jpg', 'jay02027.jpg')
    ORDER BY fe.id;
""")

rows = cursor.fetchall()


# =========================
# Compare
# =========================

print()
print("========================================")
print("EMBEDDING COMPARISON")
print("========================================")

for embedding_id, photo_id, filename, embedding_bytes in rows:

    embedding = np.frombuffer(
        embedding_bytes,
        dtype=np.float32
    )

    score = cosine_similarity(
        query_embedding,
        embedding
    )

    print(
        f"Embedding ID: {embedding_id:<4} | "
        f"Photo ID: {photo_id:<3} | "
        f"{filename:<15} | "
        f"Similarity: {score:.9f}"
    )


# =========================
# Close
# =========================

cursor.close()
conn.close()

print("========================================")
print("Done.")