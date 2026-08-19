import os
import cv2
import psycopg2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# PostgreSQL Configuration
# ==========================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


# ==========================================
# Faces To Relabel
# ==========================================

SUSPICIOUS_IDS = [
    32,
    60,
    28,
    35,
    49,
    52
]


# ==========================================
# Trusted Reference IDs
# ==========================================
# These are from SAI_5929.jpg
#
# ID 75 = Person A
# ID 76 = Person B
# ID 77 = Person C
# ID 78 = Person D

REFERENCE_IDS = {
    "A": 75,
    "B": 76,
    "C": 77,
    "D": 78
}


# ==========================================
# Load InsightFace
# ==========================================

print()
print("========================================")
print("LOADING FACE MODEL")
print("========================================")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Face model loaded.")


# ==========================================
# Database
# ==========================================

try:

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)
    exit()


# ==========================================
# Get Database Face Information
# ==========================================

def get_face_info(embedding_id):

    cursor.execute("""
        SELECT
            fe.id,
            fe.photo_id,
            fe.face_index,
            p.filename,
            p.storage_path,
            fl.person_label
        FROM face_embeddings fe

        JOIN photos p
            ON fe.photo_id = p.id

        LEFT JOIN face_labels fl
            ON fe.id = fl.embedding_id

        WHERE fe.id = %s;
    """, (embedding_id,))

    return cursor.fetchone()


# ==========================================
# Get Face Crop
# ==========================================

def get_face_crop(storage_path, face_index):

    if not os.path.exists(storage_path):

        print()
        print("Image not found:")
        print(storage_path)

        return None

    image = cv2.imread(storage_path)

    if image is None:

        print()
        print("Could not read image:")
        print(storage_path)

        return None

    faces = app.get(image)

    if len(faces) == 0:

        print()
        print("No face detected:")
        print(storage_path)

        return None

    if face_index >= len(faces):

        print()
        print(
            f"Face {face_index + 1} not found."
        )

        return None

    face = faces[face_index]

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    # Padding
    padding = 50

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


# ==========================================
# Resize Face
# ==========================================

def resize_face(image, size=280):

    return cv2.resize(
        image,
        (size, size)
    )


# ==========================================
# Create Reference Panel
# ==========================================

def create_reference_panel(reference_crops):

    size = 280

    blank = np.ones(
        (size, size, 3),
        dtype=np.uint8
    ) * 255

    panels = []

    for label in ["A", "B", "C", "D"]:

        crop = reference_crops.get(label)

        if crop is None:

            crop = blank.copy()

        else:

            crop = resize_face(
                crop,
                size
            )

        # Label area
        header = np.ones(
            (50, size, 3),
            dtype=np.uint8
        ) * 255

        cv2.putText(
            header,
            f"PERSON {label}",
            (70, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        panel = np.vstack(
            [
                header,
                crop
            ]
        )

        panels.append(panel)

    top = np.hstack(
        [
            panels[0],
            panels[1]
        ]
    )

    bottom = np.hstack(
        [
            panels[2],
            panels[3]
        ]
    )

    return np.vstack(
        [
            top,
            bottom
        ]
    )


# ==========================================
# Load Reference Faces
# ==========================================

print()
print("========================================")
print("LOADING REFERENCE FACES")
print("========================================")

reference_crops = {}

for label, embedding_id in REFERENCE_IDS.items():

    info = get_face_info(embedding_id)

    if info is None:

        print(
            f"Reference ID {embedding_id} "
            f"not found."
        )

        continue

    (
        _,
        _,
        face_index,
        filename,
        storage_path,
        current_label
    ) = info

    print(
        f"Person {label}: "
        f"ID {embedding_id} | "
        f"{filename} | "
        f"Face {face_index + 1}"
    )

    crop = get_face_crop(
        storage_path,
        face_index
    )

    reference_crops[label] = crop


# ==========================================
# Verify References
# ==========================================

missing = [
    label
    for label in ["A", "B", "C", "D"]
    if reference_crops.get(label) is None
]

if missing:

    print()
    print(
        "ERROR: Missing reference faces:"
    )

    print(
        ", ".join(missing)
    )

    cursor.close()
    conn.close()

    exit()


print()
print("All reference faces loaded.")


# ==========================================
# Relabel Suspicious Faces
# ==========================================

for number, embedding_id in enumerate(
    SUSPICIOUS_IDS,
    start=1
):

    print()
    print("========================================")
    print(
        f"FACE {number} / "
        f"{len(SUSPICIOUS_IDS)}"
    )
    print("========================================")

    info = get_face_info(embedding_id)

    if info is None:

        print(
            f"Embedding ID {embedding_id} "
            f"not found."
        )

        continue

    (
        _,
        _,
        face_index,
        filename,
        storage_path,
        current_label
    ) = info

    print(
        f"Embedding ID : {embedding_id}"
    )

    print(
        f"Photo        : {filename}"
    )

    print(
        f"Face         : {face_index + 1}"
    )

    print(
        f"Current label: {current_label}"
    )


    # --------------------------------------
    # Get Target Face
    # --------------------------------------

    target_crop = get_face_crop(
        storage_path,
        face_index
    )

    if target_crop is None:

        print(
            "Could not load target face."
        )

        continue


    # --------------------------------------
    # Target Display
    # --------------------------------------

    target_size = 300

    target_crop = resize_face(
        target_crop,
        target_size
    )

    target_header = np.ones(
        (70, target_size, 3),
        dtype=np.uint8
    ) * 255

    cv2.putText(
        target_header,
        f"TARGET ID {embedding_id}",
        (35, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    target_panel = np.vstack(
        [
            target_header,
            target_crop
        ]
    )


    # --------------------------------------
    # Reference Display
    # --------------------------------------

    reference_panel = create_reference_panel(
        reference_crops
    )

    # Resize reference panel
    reference_panel = cv2.resize(
        reference_panel,
        (600, 660)
    )


    # --------------------------------------
    # Add target beside references
    # --------------------------------------

    target_panel = cv2.resize(
        target_panel,
        (300, 370)
    )

    # White background
    canvas = np.ones(
        (660, 900, 3),
        dtype=np.uint8
    ) * 255

    # Put target
    canvas[
        145:515,
        0:300
    ] = target_panel

    # Put references
    canvas[
        0:660,
        300:900
    ] = reference_panel


    # --------------------------------------
    # Instructions
    # --------------------------------------

    cv2.putText(
        canvas,
        "A / B / C / D = Correct Person",
        (20, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        "S = Skip    Q = Quit",
        (20, 590),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Current: {current_label}",
        (20, 630),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


    window_name = (
        f"Relabel ID {embedding_id}"
    )

    cv2.imshow(
        window_name,
        canvas
    )


    # --------------------------------------
    # User Selection
    # --------------------------------------

    while True:

        key = cv2.waitKey(0) & 0xFF

        if key in [
            ord("a"),
            ord("A")
        ]:

            new_label = "A"
            break

        elif key in [
            ord("b"),
            ord("B")
        ]:

            new_label = "B"
            break

        elif key in [
            ord("c"),
            ord("C")
        ]:

            new_label = "C"
            break

        elif key in [
            ord("d"),
            ord("D")
        ]:

            new_label = "D"
            break

        elif key in [
            ord("s"),
            ord("S")
        ]:

            print(
                "Skipped."
            )

            new_label = None
            break

        elif key in [
            ord("q"),
            ord("Q")
        ]:

            print(
                "Verification stopped."
            )

            cv2.destroyAllWindows()

            cursor.close()
            conn.close()

            exit()

        else:

            print(
                "Press A, B, C, D, S or Q."
            )


    cv2.destroyAllWindows()


    # --------------------------------------
    # Update Database
    # --------------------------------------

    if new_label is not None:

        cursor.execute("""
            UPDATE face_labels
            SET person_label = %s
            WHERE embedding_id = %s;
        """, (
            new_label,
            embedding_id
        ))

        conn.commit()

        print()
        print(
            f"UPDATED: ID {embedding_id} "
            f"-> Person {new_label}"
        )


# ==========================================
# Final Summary
# ==========================================

print()
print("========================================")
print("RELABELING COMPLETE")
print("========================================")

cursor.execute("""
    SELECT
        person_label,
        COUNT(*)
    FROM face_labels
    GROUP BY person_label
    ORDER BY person_label;
""")

rows = cursor.fetchall()

for label, count in rows:

    print(
        f"Person {label}: {count} faces"
    )


cursor.close()
conn.close()

print()
print("Database connection closed.")