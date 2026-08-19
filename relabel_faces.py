import os
import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = r"G:\AI Face Photo Finder\images"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


# ============================================================
# LOAD FACE MODEL
# ============================================================

print("=" * 70)
print("FRESH FACE LABELING")
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
# DATABASE CONNECTION
# ============================================================

print()
print("Connecting to PostgreSQL...")

try:

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("PostgreSQL connected.")

except Exception as e:

    print("Database connection failed:")
    print(e)
    raise SystemExit


# ============================================================
# GET PHOTOS
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
)

files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(extensions)
])


print()
print(f"Photos found: {len(files)}")


if not files:

    print("No images found.")
    cursor.close()
    conn.close()
    raise SystemExit


# ============================================================
# LABELING LOOP
# ============================================================

total_faces = 0
saved_faces = 0
skipped_faces = 0


for photo_number, filename in enumerate(files, start=1):

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    print()
    print("=" * 70)
    print(
        f"PHOTO {photo_number}/{len(files)}"
    )
    print(
        f"File: {filename}"
    )
    print("=" * 70)


    image = cv2.imread(path)

    if image is None:

        print("ERROR: Could not read image.")
        continue


    faces = app.get(image)

    print(
        f"Faces detected: {len(faces)}"
    )


    if len(faces) == 0:

        print("No faces found.")
        continue


    # --------------------------------------------------------
    # Sort faces LEFT → RIGHT
    # --------------------------------------------------------

    faces = sorted(
        faces,
        key=lambda face: face.bbox[0]
    )


    for face_index, face in enumerate(faces):

        total_faces += 1


        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        print()
        print("-" * 70)

        print(
            f"File      : {filename}"
        )

        print(
            f"Face      : {face_index}"
        )

        print(
            f"Position  : x={x1}, y={y1}, "
            f"x2={x2}, y2={y2}"
        )

        print(
            f"Embedding : {len(face.embedding)} dimensions"
        )


        # ----------------------------------------------------
        # SHOW FACE
        # ----------------------------------------------------

        display = image.copy()

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        cv2.putText(
            display,
            f"FACE {face_index}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Resize for easier viewing
        max_width = 1200
        max_height = 800

        h, w = display.shape[:2]

        scale = min(
            max_width / w,
            max_height / h,
            1.0
        )

        if scale < 1.0:

            display = cv2.resize(
                display,
                (
                    int(w * scale),
                    int(h * scale)
                )
            )


        window_name = (
            f"{filename} - Face {face_index}"
        )

        cv2.imshow(
            window_name,
            display
        )

        cv2.waitKey(1)


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        while True:

            label = input(
                "\nPerson label "
                "(A/B/C/D/SKIP): "
            ).strip().upper()


            if label in (
                "A",
                "B",
                "C",
                "D",
                "SKIP"
            ):

                break


            print(
                "Invalid label. "
                "Enter A, B, C, D or SKIP."
            )


        cv2.destroyWindow(
            window_name
        )


        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        if label == "SKIP":

            skipped_faces += 1

            print(
                "Face skipped."
            )

            continue


        # ----------------------------------------------------
        # SAVE PHOTO
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM photos
            WHERE filename = %s
            """,
            (filename,)
        )

        photo_row = cursor.fetchone()


        if photo_row:

            photo_id = photo_row[0]

        else:

            cursor.execute(
                """
                INSERT INTO photos (filename)
                VALUES (%s)
                RETURNING id
                """,
                (filename,)
            )

            photo_id = cursor.fetchone()[0]


        # ----------------------------------------------------
        # SAVE EMBEDDING
        # ----------------------------------------------------

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32
        )


        cursor.execute(
            """
            INSERT INTO face_embeddings
            (
                photo_id,
                face_index,
                embedding
            )
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                photo_id,
                face_index,
                psycopg2.Binary(
                    embedding.tobytes()
                )
            )
        )


        embedding_id = cursor.fetchone()[0]


        # ----------------------------------------------------
        # SAVE LABEL
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO face_labels
            (
                embedding_id,
                person_label
            )
            VALUES (%s, %s)
            """,
            (
                embedding_id,
                label
            )
        )


        conn.commit()

        saved_faces += 1


        print()
        print(
            f"SAVED: "
            f"{filename} | "
            f"Face={face_index} | "
            f"Person={label} | "
            f"Embedding ID={embedding_id}"
        )


# ============================================================
# FINISH
# ============================================================

cv2.destroyAllWindows()

cursor.close()
conn.close()


print()
print("=" * 70)
print("FRESH LABELING COMPLETED")
print("=" * 70)

print(
    f"Photos scanned   : {len(files)}"
)

print(
    f"Faces detected   : {total_faces}"
)

print(
    f"Faces saved      : {saved_faces}"
)

print(
    f"Faces skipped    : {skipped_faces}"
)

print()
print(
    "PostgreSQL connection closed."
)

print("=" * 70)