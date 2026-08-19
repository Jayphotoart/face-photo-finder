import os
import psycopg2
import numpy as np
import cv2
import matplotlib.pyplot as plt


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}

THRESHOLD = 0.63


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def load_faces(conn):
    query = """
        SELECT
            fe.id,
            fe.embedding,
            fe.face_index,
            fl.person_label,
            p.filename,
            p.storage_path
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

    for embedding_id, embedding_bytes, face_index, label, filename, storage_path in rows:

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        ).copy()

        if embedding.shape[0] != 512:
            continue

        faces.append({
            "id": embedding_id,
            "embedding": embedding,
            "face_index": face_index,
            "label": label,
            "filename": filename,
            "storage_path": storage_path
        })

    return faces


def find_suspicious_faces(faces):
    suspicious = {}

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):

            a = faces[i]
            b = faces[j]

            score = cosine_similarity(
                a["embedding"],
                b["embedding"]
            )

            # Same person but unexpectedly low similarity
            if a["label"] == b["label"] and score < THRESHOLD:

                for face in (a, b):
                    if face["id"] not in suspicious:
                        suspicious[face["id"]] = {
                            "face": face,
                            "reason": "LOW SAME-PERSON SIMILARITY",
                            "worst_score": score
                        }
                    else:
                        suspicious[face["id"]]["worst_score"] = min(
                            suspicious[face["id"]]["worst_score"],
                            score
                        )

            # Different person but unexpectedly high similarity
            elif a["label"] != b["label"] and score >= THRESHOLD:

                for face in (a, b):
                    if face["id"] not in suspicious:
                        suspicious[face["id"]] = {
                            "face": face,
                            "reason": "HIGH DIFFERENT-PERSON SIMILARITY",
                            "worst_score": score
                        }
                    else:
                        suspicious[face["id"]]["worst_score"] = max(
                            suspicious[face["id"]]["worst_score"],
                            score
                        )

    return suspicious


def show_face(face):
    path = face["storage_path"]

    print("\n" + "=" * 70)
    print(f"Embedding ID : {face['id']}")
    print(f"Label        : {face['label']}")
    print(f"Filename     : {face['filename']}")
    print(f"Face index   : {face['face_index']}")
    print(f"Image path   : {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print("\nERROR: Image file not found:")
        print(path)
        return

    image = cv2.imread(path)

    if image is None:
        print("\nERROR: Could not read image.")
        return

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    h, w = image.shape[:2]

    # Current face_index is used as an index reference,
    # but the actual bounding box is not stored in face_embeddings.
    # Therefore show the full photo for now.
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(
        f"ID {face['id']} | "
        f"Label {face['label']} | "
        f"{face['filename']} | "
        f"Face {face['face_index']}"
    )
    plt.axis("off")
    plt.show()


def main():

    print("=" * 70)
    print("UNIQUE SUSPICIOUS FACE VIEWER")
    print("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

    faces = load_faces(conn)

    print(f"\nUsable faces: {len(faces)}")

    suspicious = find_suspicious_faces(faces)

    items = list(suspicious.values())

    # Most suspicious first
    items.sort(
        key=lambda x: x["worst_score"]
    )

    print(f"Unique suspicious faces: {len(items)}")

    print("\nSuspicious IDs:")

    for number, item in enumerate(items, start=1):

        face = item["face"]

        print(
            f"{number:3}. "
            f"ID={face['id']:<4} "
            f"Label={face['label']} "
            f"Score={item['worst_score']:.6f} "
            f"{item['reason']} "
            f"{face['filename']}"
        )

    print("\n" + "=" * 70)
    print("VIEWER")
    print("=" * 70)

    for number, item in enumerate(items, start=1):

        face = item["face"]

        print(
            f"\n[{number}/{len(items)}] "
            f"ID={face['id']} | "
            f"Label={face['label']} | "
            f"{face['filename']} | "
            f"Face={face['face_index']}"
        )

        choice = input(
            "View image? [Y]es / [S]kip / [Q]uit: "
        ).strip().upper()

        if choice == "Q":
            break

        if choice == "Y":
            show_face(face)

    conn.close()

    print("\nViewer finished.")


if __name__ == "__main__":
    main()