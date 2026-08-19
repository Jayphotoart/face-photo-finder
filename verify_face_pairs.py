import os
import cv2
import numpy as np
import psycopg2
from itertools import combinations


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto",
}


# ============================================================
# HOW MANY FACES PER PERSON
# Total = 11
# ============================================================

PERSON_COUNTS = {
    "A": 3,
    "B": 3,
    "C": 3,
    "D": 2,
}


WINDOW_NAME = "Face Pair Verification"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# LOAD LABELED FACES
# ============================================================

def load_labeled_faces(conn):

    query = """
        SELECT
            fl.embedding_id,
            fl.person_label,
            fe.photo_id,
            fe.face_index,
            fe.embedding,
            p.filename,
            p.storage_path
        FROM face_labels fl
        JOIN face_embeddings fe
            ON fl.embedding_id = fe.id
        JOIN photos p
            ON fe.photo_id = p.id
        WHERE fl.person_label IN ('A', 'B', 'C', 'D')
        ORDER BY
            fl.person_label,
            fl.embedding_id
    """

    with conn.cursor() as cur:

        cur.execute(query)

        rows = cur.fetchall()

    faces = []

    for row in rows:

        (
            embedding_id,
            person_label,
            photo_id,
            face_index,
            embedding_bytes,
            filename,
            storage_path,
        ) = row

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        ).copy()

        faces.append({
            "id": embedding_id,
            "person_label": person_label,
            "photo_id": photo_id,
            "face_index": face_index,
            "embedding": embedding,
            "filename": filename,
            "storage_path": storage_path,
        })

    return faces


# ============================================================
# SELECT REPRESENTATIVE FACES
# ============================================================

def select_representative_faces(faces):

    selected = []

    print("\nSelecting representative faces...")

    for person, required_count in PERSON_COUNTS.items():

        person_faces = [
            face
            for face in faces
            if face["person_label"] == person
        ]

        print(
            f"Person {person}: "
            f"{len(person_faces)} labeled faces available"
        )

        if len(person_faces) < required_count:

            raise RuntimeError(
                f"Person {person} has only "
                f"{len(person_faces)} labeled faces, "
                f"but {required_count} are required."
            )

        # Deterministic selection.
        # First N IDs after sorting.
        person_faces = sorted(
            person_faces,
            key=lambda x: x["id"]
        )

        chosen = person_faces[:required_count]

        selected.extend(chosen)

        print(
            f"Selected {person}: "
            f"{[x['id'] for x in chosen]}"
        )

    return selected


# ============================================================
# PRINT SELECTED FACES
# ============================================================

def print_selected_faces(faces):

    print("\n" + "=" * 60)
    print("SELECTED 11 FACES")
    print("=" * 60)

    for face in faces:

        print(
            f"ID {face['id']:>3} | "
            f"Person {face['person_label']} | "
            f"{face['filename']} | "
            f"Face {face['face_index']}"
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(face):

    storage_path = face["storage_path"]
    filename = face["filename"]

    candidates = []

    if storage_path:
        candidates.append(storage_path)

        if filename:
            candidates.append(
                os.path.join(
                    storage_path,
                    filename
                )
            )

    # filename may itself be a full path
    if filename:
        candidates.append(filename)

    checked = set()

    for path in candidates:

        if not path:
            continue

        path = os.path.normpath(path)

        if path in checked:
            continue

        checked.add(path)

        if os.path.isfile(path):

            image = cv2.imread(path)

            if image is not None:
                return image, path

    print(
        "\nWARNING: Image not found:"
    )

    print(
        f"  ID           : {face['id']}"
    )

    print(
        f"  filename     : {filename}"
    )

    print(
        f"  storage_path : {storage_path}"
    )

    return None, None


# ============================================================
# RESIZE
# ============================================================

def resize_to_height(
    image,
    target_height=600
):

    h, w = image.shape[:2]

    if h == target_height:
        return image

    scale = target_height / h

    new_width = int(w * scale)

    return cv2.resize(
        image,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# ADD LABEL
# ============================================================

def add_label(image, text):

    img = image.copy()

    bar_height = 55

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], bar_height),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        text,
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return img


# ============================================================
# PAD WIDTH
# ============================================================

def pad_width(
    image,
    target_width
):

    difference = target_width - image.shape[1]

    if difference <= 0:
        return image

    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        difference,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )


# ============================================================
# CREATE SIDE-BY-SIDE IMAGE
# ============================================================

def create_display(
    face_a,
    face_b,
    similarity,
    pair_number,
    total_pairs
):

    image_a, _ = load_image(face_a)
    image_b, _ = load_image(face_b)

    if image_a is None or image_b is None:
        return None

    image_a = resize_to_height(
        image_a,
        600
    )

    image_b = resize_to_height(
        image_b,
        600
    )

    label_a = (
        f"A | ID {face_a['id']} | "
        f"{face_a['filename']} | "
        f"Face {face_a['face_index']}"
    )

    label_b = (
        f"B | ID {face_b['id']} | "
        f"{face_b['filename']} | "
        f"Face {face_b['face_index']}"
    )

    image_a = add_label(
        image_a,
        label_a
    )

    image_b = add_label(
        image_b,
        label_b
    )

    max_width = max(
        image_a.shape[1],
        image_b.shape[1]
    )

    image_a = pad_width(
        image_a,
        max_width
    )

    image_b = pad_width(
        image_b,
        max_width
    )

    pair_image = np.hstack(
        (image_a, image_b)
    )

    # --------------------------------------------------------
    # INFORMATION BAR
    # --------------------------------------------------------

    bar_height = 95

    info = np.zeros(
        (
            bar_height,
            pair_image.shape[1],
            3
        ),
        dtype=np.uint8
    )

    cv2.putText(
        info,
        f"PAIR {pair_number}/{total_pairs}   "
        f"Similarity: {similarity:.6f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        info,
        "Y = SAME PERSON    "
        "N = DIFFERENT PERSON    "
        "S = SKIP    Q = QUIT",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return np.vstack(
        (
            info,
            pair_image
        )
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        (norm_a * norm_b)
    )


# ============================================================
# EXISTING RESULT
# ============================================================

def get_existing_result(
    conn,
    face_id_a,
    face_id_b
):

    query = """
        SELECT result
        FROM verification_results
        WHERE face_id_a = %s
          AND face_id_b = %s
    """

    with conn.cursor() as cur:

        cur.execute(
            query,
            (
                face_id_a,
                face_id_b
            )
        )

        row = cur.fetchone()

    if row:
        return row[0]

    return None


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    conn,
    face_id_a,
    face_id_b,
    similarity,
    result
):

    query = """
        INSERT INTO verification_results
        (
            face_id_a,
            face_id_b,
            similarity,
            result
        )
        VALUES (%s, %s, %s, %s)

        ON CONFLICT (face_id_a, face_id_b)
        DO UPDATE SET
            similarity = EXCLUDED.similarity,
            result = EXCLUDED.result,
            created_at = CURRENT_TIMESTAMP
    """

    with conn.cursor() as cur:

        cur.execute(
            query,
            (
                face_id_a,
                face_id_b,
                similarity,
                result
            )
        )

    conn.commit()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FACE PAIR VERIFICATION")
    print("=" * 60)

    print(
        "\nAutomatic selection:"
    )

    print(
        "A = 3 faces"
    )

    print(
        "B = 3 faces"
    )

    print(
        "C = 3 faces"
    )

    print(
        "D = 2 faces"
    )

    print(
        "Total = 11 faces"
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:

        conn = get_connection()

        print(
            "\nPostgreSQL connection successful!"
        )

    except Exception as e:

        print(
            "\nDatabase connection failed:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # LOAD LABELS
    # --------------------------------------------------------

    try:

        all_faces = load_labeled_faces(
            conn
        )

        print(
            f"\nTotal labeled faces: "
            f"{len(all_faces)}"
        )

    except Exception as e:

        print(
            "\nError loading labeled faces:"
        )

        print(e)

        conn.close()

        return

    # --------------------------------------------------------
    # SELECT 11
    # --------------------------------------------------------

    try:

        selected_faces = (
            select_representative_faces(
                all_faces
            )
        )

    except Exception as e:

        print(
            "\nSelection error:"
        )

        print(e)

        conn.close()

        return

    print_selected_faces(
        selected_faces
    )

    # --------------------------------------------------------
    # CREATE 55 PAIRS
    # --------------------------------------------------------

    pairs = list(
        combinations(
            selected_faces,
            2
        )
    )

    total_pairs = len(pairs)

    print(
        f"\nTotal unique pairs: "
        f"{total_pairs}"
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    completed = 0

    for pair_number, (
        face_a,
        face_b
    ) in enumerate(
        pairs,
        start=1
    ):

        id_a = face_a["id"]
        id_b = face_b["id"]

        # ----------------------------------------------------
        # EXISTING RESULT
        # ----------------------------------------------------

        existing = get_existing_result(
            conn,
            id_a,
            id_b
        )

        if existing is not None:

            print(
                f"\n[{pair_number}/{total_pairs}] "
                f"Already verified: "
                f"{id_a} vs {id_b} = "
                f"{existing}"
            )

            completed += 1

            continue

        # ----------------------------------------------------
        # SIMILARITY
        # ----------------------------------------------------

        similarity = cosine_similarity(
            face_a["embedding"],
            face_b["embedding"]
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display = create_display(
            face_a,
            face_b,
            similarity,
            pair_number,
            total_pairs
        )

        if display is None:

            print(
                f"\nSkipping pair "
                f"{id_a} vs {id_b}"
            )

            continue

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        print("\n" + "=" * 60)

        print(
            f"PAIR {pair_number}/{total_pairs}"
        )

        print(
            f"A: ID {id_a} | "
            f"Person label: "
            f"{face_a['person_label']} | "
            f"{face_a['filename']} | "
            f"Face {face_a['face_index']}"
        )

        print(
            f"B: ID {id_b} | "
            f"Person label: "
            f"{face_b['person_label']} | "
            f"{face_b['filename']} | "
            f"Face {face_b['face_index']}"
        )

        print(
            f"Cosine similarity: "
            f"{similarity:.6f}"
        )

        print(
            "Y = SAME PERSON"
            " | N = DIFFERENT PERSON"
            " | S = SKIP"
            " | Q = QUIT"
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key in (
                ord("y"),
                ord("Y")
            ):

                result = "Y"

                break

            elif key in (
                ord("n"),
                ord("N")
            ):

                result = "N"

                break

            elif key in (
                ord("s"),
                ord("S")
            ):

                result = "S"

                break

            elif key in (
                ord("q"),
                ord("Q"),
                27
            ):

                print(
                    "\nVerification stopped."
                )

                cv2.destroyAllWindows()

                conn.close()

                return

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_result(
            conn,
            id_a,
            id_b,
            similarity,
            result
        )

        completed += 1

        print(
            f"Saved: "
            f"{id_a} vs {id_b} "
            f"=> {result}"
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    cv2.destroyAllWindows()

    conn.close()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    print(
        f"Processed: "
        f"{completed}/{total_pairs}"
    )

    print(
        "\nResults saved in:"
    )

    print(
        "verification_results"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()