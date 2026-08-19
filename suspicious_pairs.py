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

THRESHOLD = 0.63


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def main():

    print("=" * 70)
    print("SUSPICIOUS FACE PAIR ANALYSIS")
    print("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            fe.id,
            fe.embedding,
            fl.person_label,
            p.filename,
            fe.face_index
        FROM face_embeddings fe
        JOIN face_labels fl
            ON fe.id = fl.embedding_id
        JOIN photos p
            ON fe.photo_id = p.id
        WHERE fl.person_label IN ('A', 'B', 'C', 'D')
        ORDER BY fe.id;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    faces = []

    for embedding_id, embedding_bytes, label, filename, face_index in rows:

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        ).copy()

        if embedding.shape[0] != 512:
            print(
                f"WARNING: ID {embedding_id} "
                f"has dimension {embedding.shape[0]}"
            )
            continue

        faces.append({
            "id": embedding_id,
            "label": label,
            "filename": filename,
            "face_index": face_index,
            "embedding": embedding
        })

    print(f"\nUsable faces: {len(faces)}")

    same_low = []
    different_high = []

    for a, b in combinations(faces, 2):

        score = cosine_similarity(
            a["embedding"],
            b["embedding"]
        )

        if a["label"] == b["label"]:

            if score < THRESHOLD:
                same_low.append(
                    (score, a, b)
                )

        else:

            if score >= THRESHOLD:
                different_high.append(
                    (score, a, b)
                )

    # --------------------------------------------------
    # SAME PERSON BUT LOW SIMILARITY
    # --------------------------------------------------

    same_low.sort(key=lambda x: x[0])

    print("\n" + "=" * 70)
    print(
        f"SAME PERSON BUT LOW SIMILARITY "
        f"(< {THRESHOLD})"
    )
    print("=" * 70)

    print(f"Total: {len(same_low)}")

    for score, a, b in same_low:

        print(
            f"\nSimilarity: {score:.6f}"
        )

        print(
            f"  {a['label']} | "
            f"ID {a['id']} | "
            f"{a['filename']} | "
            f"Face {a['face_index']}"
        )

        print(
            f"  {b['label']} | "
            f"ID {b['id']} | "
            f"{b['filename']} | "
            f"Face {b['face_index']}"
        )

    # --------------------------------------------------
    # DIFFERENT PERSON BUT HIGH SIMILARITY
    # --------------------------------------------------

    different_high.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\n" + "=" * 70)
    print(
        f"DIFFERENT PERSON BUT HIGH SIMILARITY "
        f"(>= {THRESHOLD})"
    )
    print("=" * 70)

    print(f"Total: {len(different_high)}")

    for score, a, b in different_high:

        print(
            f"\nSimilarity: {score:.6f}"
        )

        print(
            f"  {a['label']} | "
            f"ID {a['id']} | "
            f"{a['filename']} | "
            f"Face {a['face_index']}"
        )

        print(
            f"  {b['label']} | "
            f"ID {b['id']} | "
            f"{b['filename']} | "
            f"Face {b['face_index']}"
        )

    conn.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()