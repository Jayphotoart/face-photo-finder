
import cv2
import numpy as np
import psycopg2
from collections import defaultdict
from insightface.app import FaceAnalysis

# ============================================================
# MATCH DECISION THRESHOLDS
# ============================================================

STRONG_THRESHOLD = 0.55
LIKELY_THRESHOLD = 0.40
REVIEW_THRESHOLD = 0.35


def classify_similarity(similarity):
    if similarity >= STRONG_THRESHOLD:
        return "STRONG MATCH"
    elif similarity >= LIKELY_THRESHOLD:
        return "LIKELY MATCH"
    elif similarity >= REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "NO MATCH"


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

IMAGE_PATH = r"test_images	est9.jpg"

# Database comparison threshold
MATCH_THRESHOLD = 0.25

# Number of database matches to display for each query face
TOP_MATCHES_PER_FACE = 5

# Only these labels are real persons
VALID_LABELS = ("A", "B", "C", "D")


# ============================================================
# LOAD FACE MODEL
# ============================================================

print("=" * 70)
print("AI FACE SEARCH - FACE LEVEL + PERSON LEVEL")
print("=" * 70)

print()
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


# ============================================================
# LOAD QUERY IMAGE
# ============================================================

print()
print("Query image:", IMAGE_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found:", IMAGE_PATH)
    raise SystemExit

print("Image loaded.")


# ============================================================
# DETECT QUERY FACES
# ============================================================

faces = app.get(image)

print("Faces detected:", len(faces))

if len(faces) == 0:
    print("No face found.")
    raise SystemExit


# ============================================================
# CREATE QUERY EMBEDDINGS
# ============================================================

query_embeddings = []

for i, face in enumerate(faces):

    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        continue

    embedding = embedding / norm

    query_embeddings.append({
        "query_face": i,
        "embedding": embedding
    })


print(
    "Usable query faces:",
    len(query_embeddings)
)


# ============================================================
# CONNECT DATABASE
# ============================================================

try:

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)
    raise SystemExit


# ============================================================
# GET LABELED EMBEDDINGS
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

print("Database labeled faces:", len(rows))

if len(rows) == 0:

    print()
    print("ERROR: No labeled faces found in database.")

    cursor.close()
    conn.close()

    raise SystemExit


# ============================================================
# DATA STRUCTURES
# ============================================================

# All useful database matches for each query face
query_face_matches = defaultdict(list)

# Best match of each query face against each person
# key = (query_face, person)
best_query_person_match = {}

# Person-level collection
person_matches = defaultdict(list)


# ============================================================
# COMPARE EACH QUERY FACE WITH DATABASE
# ============================================================

for query_data in query_embeddings:

    query_index = query_data["query_face"]
    query_embedding = query_data["embedding"]

    for row in rows:

        embedding_id = row[0]
        photo_id = row[1]
        face_index = row[2]
        embedding_bytes = row[3]
        filename = row[4]
        person_label = row[5]

        # ----------------------------------------------------
        # Convert PostgreSQL BYTEA / memoryview to numpy
        # ----------------------------------------------------

        stored_embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        )

        if stored_embedding.size != query_embedding.size:
            continue

        stored_norm = np.linalg.norm(stored_embedding)

        if stored_norm == 0:
            continue

        stored_embedding = (
            stored_embedding / stored_norm
        )

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        similarity = float(
            np.dot(
                query_embedding,
                stored_embedding
            )
        )

        if similarity < MATCH_THRESHOLD:
            continue

        match = {

            "query_face": query_index,

            "embedding_id": embedding_id,

            "photo_id": photo_id,

            "face_index": face_index,

            "filename": filename,

            "person": person_label,

            "similarity": similarity
        }

        # All matches for this query face
        query_face_matches[query_index].append(match)

        # ----------------------------------------------------
        # Keep only BEST database match for this
        # query-face + person combination
        # ----------------------------------------------------

        key = (
            query_index,
            person_label
        )

        current_best = best_query_person_match.get(key)

        if (
            current_best is None
            or similarity > current_best["similarity"]
        ):
            best_query_person_match[key] = match


# ============================================================
# BUILD PERSON MATCHES
# ============================================================

for key, match in best_query_person_match.items():

    person_label = match["person"]

    person_matches[person_label].append(match)


# ============================================================
# SORT QUERY FACE MATCHES
# ============================================================

for query_index in query_face_matches:

    query_face_matches[query_index].sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


# ============================================================
# ============================================================
# LEVEL 1 - QUERY FACE LEVEL
# ============================================================
# ============================================================

print()
print("=" * 70)
print("QUERY FACE LEVEL MATCHING")
print("=" * 70)


query_face_best_results = []


for query_data in query_embeddings:

    query_index = query_data["query_face"]

    matches = query_face_matches.get(
        query_index,
        []
    )

    print()
    print(
        f"QUERY FACE {query_index}"
    )

    print("-" * 70)

    if not matches:

        print("No database match crossed threshold.")

        query_face_best_results.append({

            "query_face": query_index,

            "best_person": None,

            "best_similarity": 0.0,

            "best_match": None
        })

        continue


    # --------------------------------------------------------
    # Find best person for this query face
    # --------------------------------------------------------

    best_match = matches[0]

    print(
        f"Best Person : {best_match['person']}"
    )

    decision = classify_similarity(
        best_match["similarity"]
    )

    print(
        f"Similarity  : "
        f"{best_match['similarity']:.4f}"
    )

    print(
        f"Decision    : "
        f"{decision}"
    )

    print(
        f"Photo       : "
        f"{best_match['filename']}"
    )

    print(
        f"Face        : "
        f"{best_match['face_index']}"
    )

    print()
    print("Top Database Matches:")

    for rank, match in enumerate(
        matches[:TOP_MATCHES_PER_FACE],
        start=1
    ):

        print(
            f"   {rank}. "
            f"Person {match['person']}"
            f" | Similarity={match['similarity']:.4f}"
            f" | {match['filename']}"
            f" | Face={match['face_index']}"
        )


    query_face_best_results.append({

        "query_face": query_index,

        "best_person": best_match["person"],

        "best_similarity": best_match["similarity"],

        "best_match": best_match
    })


# ============================================================
# ============================================================
# LEVEL 2 - PERSON LEVEL AGGREGATION
# ============================================================
# IMPORTANT:
#
# Only the BEST PERSON for each query face is allowed
# to contribute to person-level scoring.
#
# Example:
#
# Face 1:
#   C = 0.7990  <-- winner
#   A = 0.7223
#
# Only C is counted.
# A's 0.7223 is NOT counted as support for Person A.
# ============================================================
# ============================================================

person_matches = defaultdict(list)


for result in query_face_best_results:

    query_face = result["query_face"]

    best_person = result["best_person"]

    best_similarity = result["best_similarity"]

    best_match = result["best_match"]


    # No usable match for this query face
    if best_person is None:
        continue


    # --------------------------------------------------------
    # ONLY WINNING PERSON IS ADDED
    # --------------------------------------------------------

    person_matches[best_person].append({

        "query_face": query_face,

        "embedding_id":
            best_match["embedding_id"],

        "photo_id":
            best_match["photo_id"],

        "face_index":
            best_match["face_index"],

        "filename":
            best_match["filename"],

        "person":
            best_person,

        "similarity":
            best_similarity
    })


# ============================================================
# BUILD PERSON RESULTS
# ============================================================

person_results = []


for person_label in VALID_LABELS:

    matches = person_matches.get(
        person_label,
        []
    )

    if not matches:
        continue


    # Highest similarity first
    matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    similarities = [
        x["similarity"]
        for x in matches
    ]


    # --------------------------------------------------------
    # Best similarity for this person
    # --------------------------------------------------------

    best_similarity = max(
        similarities
    )


    # --------------------------------------------------------
    # Average similarity of ONLY winning query faces
    # --------------------------------------------------------

    average_similarity = float(
        np.mean(similarities)
    )


    # --------------------------------------------------------
    # Number of query faces that selected this person
    # --------------------------------------------------------

    supporting_query_faces = len(
        set(
            x["query_face"]
            for x in matches
        )
    )


    # --------------------------------------------------------
    # Since each query face contributes only once,
    # this is effectively the number of supporting faces.
    # --------------------------------------------------------

    supporting_database_matches = len(
        matches
    )


    # --------------------------------------------------------
    # Person Score
    # --------------------------------------------------------

    person_score = (
        best_similarity * 0.70
        +
        average_similarity * 0.30
    )


    person_results.append({

        "person": person_label,

        "best_similarity":
            best_similarity,

        "average_similarity":
            average_similarity,

        "supporting_query_faces":
            supporting_query_faces,

        "supporting_database_matches":
            supporting_database_matches,

        "score":
            person_score,

        "matches":
            matches
    })


# ============================================================
# SORT PERSONS
# ============================================================

person_results.sort(
    key=lambda x: x["score"],
    reverse=True
)



# ============================================================
# DISPLAY PERSON RESULTS
# ============================================================

print()
print("=" * 70)
print("PERSON LEVEL AGGREGATION")
print("=" * 70)


if len(person_results) == 0:

    print()
    print(
        "No person crossed threshold:",
        MATCH_THRESHOLD
    )

else:

    for rank, person in enumerate(
        person_results,
        start=1
    ):

        print()
        print(
            f"{rank}. PERSON {person['person']}"
        )

        print(
            f"   Person Score              : "
            f"{person['score']:.4f}"
        )

        print(
            f"   Best Similarity           : "
            f"{person['best_similarity']:.4f}"
        )

        print(
            f"   Average Similarity        : "
            f"{person['average_similarity']:.4f}"
        )

        print(
            f"   Supporting Query Faces    : "
            f"{person['supporting_query_faces']}"
        )

        print(
            f"   Supporting DB Matches     : "
            f"{person['supporting_database_matches']}"
        )

        print()
        print("   Best Supporting Matches:")

        for match in person["matches"]:

            print(
                f"      - QueryFace="
                f"{match['query_face']}"
                f" | {match['filename']}"
                f" | Face={match['face_index']}"
                f" | Similarity="
                f"{match['similarity']:.4f}"
            )


# ============================================================
# ============================================================
# QUERY FACE → PERSON SUMMARY
# ============================================================
# ============================================================

print()
print("=" * 70)
print("QUERY FACE → PERSON SUMMARY")
print("=" * 70)


for result in query_face_best_results:

    query_face = result["query_face"]

    best_person = result["best_person"]

    best_similarity = result["best_similarity"]


    if best_person is None:

        print(
            f"Face {query_face} → NO MATCH"
        )

    else:

        print(
            f"Face {query_face} "
            f"→ Person {best_person}"
            f" | Similarity={best_similarity:.4f}"
        )


# ============================================================
# CONFIDENCE DECISION
# ============================================================

def get_confidence(
    person,
    second_person=None
):

    best = person["best_similarity"]

    avg = person["average_similarity"]

    support_faces = person[
        "supporting_query_faces"
    ]


    # --------------------------------------------------------
    # IMPORTANT:
    # Margin is calculated using PERSON SCORE
    # --------------------------------------------------------

    if second_person is not None:

        second_score = second_person["score"]

        margin = (
            person["score"]
            -
            second_score
        )

    else:

        second_score = 0.0

        margin = person["score"]


    # --------------------------------------------------------
    # STRONG MATCH
    # --------------------------------------------------------

    if (
        best >= 0.70
        and avg >= 0.70
        and support_faces >= 1
        and margin >= 0.10
    ):

        return "STRONG MATCH", margin


    # --------------------------------------------------------
    # LIKELY MATCH
    # --------------------------------------------------------

    if (
        best >= 0.60
        and avg >= 0.55
        and support_faces >= 1
        and margin >= 0.08
    ):

        return "LIKELY MATCH", margin


    # --------------------------------------------------------
    # WEAK / REVIEW
    # --------------------------------------------------------

    if best >= MATCH_THRESHOLD:

        return "WEAK / REVIEW", margin


    # --------------------------------------------------------
    # NO MATCH
    # --------------------------------------------------------

    return "NO MATCH", margin


# ============================================================
# FINAL CONFIDENCE DECISION
# ============================================================

print()
print("=" * 70)
print("FINAL CONFIDENCE DECISION")
print("=" * 70)


if not person_results:

    print()
    print("Decision : NO MATCH")
    print(
        "Reason   : No person crossed threshold."
    )

else:

    best = person_results[0]

    if len(person_results) >= 2:
        second = person_results[1]
    else:
        second = None

    decision = classify_similarity(
        best["score"]
    )

    if second is not None:
        margin = (
            best["score"]
            - second["score"]
        )
    else:
        margin = best["score"]

    print()

    print(
        f"Top Person              : "
        f"{best['person']}"
    )

    print(
        f"Best Similarity          : "
        f"{best['best_similarity']:.4f}"
    )

    print(
        f"Average Similarity       : "
        f"{best['average_similarity']:.4f}"
    )

    print(
        f"Supporting Query Faces   : "
        f"{best['supporting_query_faces']}"
    )

    print(
        f"Supporting DB Matches    : "
        f"{best['supporting_database_matches']}"
    )

    print(
        f"Person Score             : "
        f"{best['score']:.4f}"
    )

    print(
        f"Person Score Margin      : "
        f"{margin:.4f}"
    )

    if second is not None:

        print(
            f"Second Person            : "
            f"{second['person']}"
        )

        print(
            f"Second Person Score      : "
            f"{second['score']:.4f}"
        )

    print()

    print(
        f"FINAL DECISION           : "
        f"{decision}"
    )


# ============================================================
# BEST PERSON MATCHED PHOTOS
# ============================================================

if person_results:

    print()
    print("=" * 70)
    print("BEST PERSON MATCHED PHOTOS")
    print("=" * 70)


    best = person_results[0]


    seen_photos = set()


    for match in best["matches"]:

        photo_id = match["photo_id"]


        if photo_id in seen_photos:
            continue


        seen_photos.add(photo_id)


        print(
            f"{match['filename']}"
            f" | QueryFace="
            f"{match['query_face']}"
            f" | DB Face="
            f"{match['face_index']}"
            f" | Similarity="
            f"{match['similarity']:.4f}"
        )


# ============================================================
# CLOSE DATABASE
# ============================================================

cursor.close()

conn.close()


print()
print("=" * 70)
print("PostgreSQL connection closed.")
print("Search completed.")
print("=" * 70)

