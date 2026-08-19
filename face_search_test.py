import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

IMAGE_PATH = "test2.jpg"

# Based on current threshold analysis
THRESHOLD = 0.25

# Maximum results to display
TOP_K = 20


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("=" * 70)
print("AI FACE SEARCH")
print("=" * 70)

print("\nLoading face model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Face model loaded.")


# ============================================================
# READ QUERY IMAGE
# ============================================================

print("\nQuery image:", IMAGE_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found:", IMAGE_PATH)
    exit()

print("Image loaded.")


# ============================================================
# DETECT FACES
# ============================================================

faces = app.get(image)

print("Faces detected:", len(faces))

if len(faces) == 0:
    print("ERROR: No face found.")
    exit()


# ============================================================
# USE FIRST FACE
# ============================================================

query_face = faces[0]

query_embedding = np.asarray(
    query_face.embedding,
    dtype=np.float32
)

# Normalize query embedding
query_embedding = query_embedding / np.linalg.norm(query_embedding)


# ============================================================
# CONNECT POSTGRESQL
# ============================================================

try:

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)

    exit()


# ============================================================
# GET DATABASE EMBEDDINGS
# ============================================================

cursor.execute("""
    SELECT
        fe.id,
        fe.photo_id,
        fe.face_index,
        fe.embedding,
        p.filename
    FROM face_embeddings fe
    JOIN photos p
        ON p.id = fe.photo_id
    ORDER BY fe.id;
""")

rows = cursor.fetchall()

print("Database embeddings:", len(rows))


# ============================================================
# COMPARE QUERY WITH DATABASE
# ============================================================

results = []

for row in rows:

    embedding_id = row[0]
    photo_id = row[1]
    face_index = row[2]
    embedding_bytes = row[3]
    filename = row[4]

    try:

        stored_embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        )

        # Safety check
        if stored_embedding.shape != query_embedding.shape:
            print(
                f"Skipping ID {embedding_id}: "
                f"embedding dimension mismatch"
            )
            continue

        # Normalize stored embedding
        norm = np.linalg.norm(stored_embedding)

        if norm == 0:
            continue

        stored_embedding = stored_embedding / norm

        # Cosine similarity
        similarity = float(
            np.dot(
                query_embedding,
                stored_embedding
            )
        )

        results.append({
            "embedding_id": embedding_id,
            "photo_id": photo_id,
            "filename": filename,
            "face_index": face_index,
            "similarity": similarity
        })

    except Exception as e:

        print(
            f"Error processing embedding "
            f"{embedding_id}: {e}"
        )


# ============================================================
# SORT BY SIMILARITY
# ============================================================

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# ============================================================
# FILTER MATCHES
# ============================================================

matches = [
    result
    for result in results
    if result["similarity"] >= THRESHOLD
]


# ============================================================
# DISPLAY ALL TOP RESULTS
# ============================================================

print()
print("=" * 70)
print("TOP SIMILARITY RESULTS")
print("=" * 70)

for i, result in enumerate(results[:TOP_K], start=1):

    status = (
        "MATCH"
        if result["similarity"] >= THRESHOLD
        else "NO MATCH"
    )

    print(
        f"{i:2}. "
        f"{result['filename']:<30} "
        f"Face={result['face_index']}  "
        f"Similarity={result['similarity']:.4f}  "
        f"{status}"
    )


# ============================================================
# DISPLAY MATCHES ONLY
# ============================================================

print()
print("=" * 70)
print(f"MATCHES (Threshold >= {THRESHOLD})")
print("=" * 70)

if len(matches) == 0:

    print("No matching face found.")

else:

    for i, result in enumerate(matches[:TOP_K], start=1):

        print(
            f"{i}. {result['filename']}"
        )

        print(
            f"   Photo ID   : {result['photo_id']}"
        )

        print(
            f"   Embedding  : {result['embedding_id']}"
        )

        print(
            f"   Face Index : {result['face_index']}"
        )

        print(
            f"   Similarity : {result['similarity']:.4f}"
        )

        print(
            f"   Status     : MATCH"
        )

        print("-" * 70)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SEARCH SUMMARY")
print("=" * 70)

print(
    f"Query image       : {IMAGE_PATH}"
)

print(
    f"Faces detected    : {len(faces)}"
)

print(
    f"Database faces    : {len(rows)}"
)

print(
    f"Threshold         : {THRESHOLD}"
)

print(
    f"Matching faces    : {len(matches)}"
)

print("=" * 70)


# ============================================================
# CLOSE DATABASE
# ============================================================

cursor.close()
conn.close()

print("PostgreSQL connection closed.")
print("Search completed.")