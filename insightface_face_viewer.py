import os
import cv2
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis


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

            # Same person but low similarity
            if a["label"] == b["label"] and score < THRESHOLD:

                for face in (a, b):

                    if face["id"] not in suspicious:

                        suspicious[face["id"]] = {
                            "face": face,
                            "min_same_score": score,
                            "max_different_score": None
                        }

                    else:

                        current = suspicious[face["id"]]["min_same_score"]

                        if current is None:
                            suspicious[face["id"]]["min_same_score"] = score
                        else:
                            suspicious[face["id"]]["min_same_score"] = min(
                                current,
                                score
                            )

            # Different person but high similarity
            elif a["label"] != b["label"] and score >= THRESHOLD:

                for face in (a, b):

                    if face["id"] not in suspicious:

                        suspicious[face["id"]] = {
                            "face": face,
                            "min_same_score": None,
                            "max_different_score": score
                        }

                    else:

                        current = suspicious[face["id"]]["max_different_score"]

                        if current is None:
                            suspicious[face["id"]]["max_different_score"] = score
                        else:
                            suspicious[face["id"]]["max_different_score"] = max(
                                current,
                                score
                            )

    return suspicious

def detect_faces(image, app):

    faces = app.get(image)

    result = []

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        result.append({
            "bbox": (x1, y1, x2, y2),
            "det_score": float(face.det_score)
        })

    return result


def show_exact_face(face_info, app):

    path = face_info["storage_path"]

    print("\n" + "=" * 70)
    print(f"Embedding ID : {face_info['id']}")
    print(f"Current Label: {face_info['label']}")
    print(f"Filename     : {face_info['filename']}")
    print(f"Face index   : {face_info['face_index']}")
    print(f"Image path   : {path}")
    print("=" * 70)

    if not os.path.exists(path):

        print("\nERROR: Image not found:")
        print(path)

        return

    image = cv2.imread(path)

    if image is None:

        print("\nERROR: Could not read image.")

        return

    detected = detect_faces(
        image,
        app
    )

    if not detected:

        print("\nERROR: InsightFace found no faces.")

        return

    face_index = face_info["face_index"]

    if face_index < 0 or face_index >= len(detected):

        print(
            f"\nERROR: Stored face_index={face_index}, "
            f"but InsightFace detected {len(detected)} faces."
        )

        print("\nDetected faces:")

        for i, item in enumerate(detected):

            print(
                f"Face {i}: "
                f"bbox={item['bbox']} "
                f"score={item['det_score']:.4f}"
            )

        return

    x1, y1, x2, y2 = detected[face_index]["bbox"]

    h, w = image.shape[:2]

    # Padding around face
    padding = 40

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(crop_rgb)

    plt.title(
        f"ID {face_info['id']} | "
        f"Label {face_info['label']} | "
        f"{face_info['filename']} | "
        f"Face {face_index}"
    )

    plt.axis("off")

    plt.show()


def main():

    print("=" * 70)
    print("INSIGHTFACE SUSPICIOUS FACE VIEWER")
    print("=" * 70)

    print("\nLoading InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print("InsightFace loaded.")

    conn = psycopg2.connect(**DB_CONFIG)

    faces = load_faces(conn)

    print(f"\nUsable faces: {len(faces)}")

    suspicious = find_suspicious_faces(faces)

    items = list(suspicious.values())

    print(
        f"Unique suspicious faces: {len(items)}"
    )

    # Sort suspicious faces
    def sort_score(item):
        same_score = item["min_same_score"]
        different_score = item["max_different_score"]

        if same_score is not None:
            return same_score

        if different_score is not None:
            return -different_score

        return 0

    items.sort(key=sort_score)
    print("\nSuspicious IDs:")

    for number, item in enumerate(items, start=1):

        face = item["face"]

        print(
            f"{number:3}. "
            f"ID={face['id']:<4} "
            f"Label={face['label']} "
            f"{face['filename']} "
            f"Face={face['face_index']}"
        )

    for number, item in enumerate(items, start=1):

        face = item["face"]

        print("\n" + "=" * 70)

        print(
            f"[{number}/{len(items)}] "
            f"ID={face['id']} | "
            f"Label={face['label']} | "
            f"{face['filename']} | "
            f"Face={face['face_index']}"
        )

        choice = input(
            "View exact face? [Y]es / [S]kip / [Q]uit: "
        ).strip().upper()

        if choice == "Q":

            break

        if choice == "Y":

            show_exact_face(
                face,
                app
            )

    conn.close()

    print("\nViewer finished.")


if __name__ == "__main__":
    main()