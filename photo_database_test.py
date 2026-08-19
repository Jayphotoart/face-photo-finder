import os
import cv2
import numpy as np
import psycopg2
from insightface.app import FaceAnalysis


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "face_finder",
    "user": "postgres",
    "password": "Jayphoto"
}


# ============================================================
# IMAGES FOLDER
# ============================================================

IMAGE_FOLDER = "images"


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

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

try:
    conn = psycopg2.connect(**DB_CONFIG)

    print("PostgreSQL connection successful!")

except Exception as e:

    print("Database connection failed:")
    print(e)

    raise SystemExit


# ============================================================
# CHECK IMAGES FOLDER
# ============================================================

if not os.path.exists(IMAGE_FOLDER):

    print(f"Folder not found: {IMAGE_FOLDER}")

    conn.close()

    raise SystemExit


# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = []

for file_name in sorted(os.listdir(IMAGE_FOLDER)):

    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_files.append(file_name)


print()
print(f"Images found: {len(image_files)}")
print()


# ============================================================
# PROCESS IMAGES
# ============================================================

total_photos = 0
total_faces = 0


for file_name in image_files:

    image_path = os.path.join(
        IMAGE_FOLDER,
        file_name
    )

    print("=" * 60)
    print(f"Processing: {file_name}")

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("Could not read image.")
        continue


    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    detected_faces = app.get(image)

    print(
        f"Faces detected: "
        f"{len(detected_faces)}"
    )


    if len(detected_faces) == 0:

        print("No face found. Skipping.")

        continue


    # --------------------------------------------------------
    # SAVE PHOTO
    # --------------------------------------------------------

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO photos
                (
                    filename,
                    storage_path
                )
                VALUES (%s, %s)
                RETURNING id;
                """,
                (
                    file_name,
                    image_path
                )
            )

            photo_id = cursor.fetchone()[0]


            # ------------------------------------------------
            # SAVE EVERY DETECTED FACE
            # ------------------------------------------------

            for face_index, face in enumerate(
                detected_faces
            ):

                # --------------------------------------------
                # INSERT INTO faces
                # --------------------------------------------

                cursor.execute(
                    """
                    INSERT INTO faces
                    (
                        photo_id,
                        face_index
                    )
                    VALUES (%s, %s)
                    RETURNING id;
                    """,
                    (
                        photo_id,
                        face_index
                    )
                )

                face_id = cursor.fetchone()[0]


                # --------------------------------------------
                # GET EMBEDDING
                # --------------------------------------------

                embedding = np.asarray(
                    face.embedding,
                    dtype=np.float32
                )


                print(
                    f"Face {face_index + 1}: "
                    f"faces.id={face_id}, "
                    f"embedding shape={embedding.shape}"
                )


                # --------------------------------------------
                # CONVERT EMBEDDING TO BYTEA
                # --------------------------------------------

                embedding_bytes = embedding.tobytes()


                # --------------------------------------------
                # SAVE EMBEDDING
                # --------------------------------------------

                cursor.execute(
                    """
                    INSERT INTO face_embeddings
                    (
                        photo_id,
                        face_index,
                        embedding
                    )
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        photo_id,
                        face_index,
                        psycopg2.Binary(
                            embedding_bytes
                        )
                    )
                )

                embedding_id = cursor.fetchone()[0]


                print(
                    f"              "
                    f"embedding_id={embedding_id}"
                )

                total_faces += 1


        # Commit this photo
        conn.commit()

        total_photos += 1

        print(
            f"Saved photo ID: {photo_id}"
        )


    except Exception as e:

        conn.rollback()

        print(
            f"ERROR processing {file_name}:"
        )

        print(e)

        continue


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("IMPORT COMPLETE")
print("=" * 60)

print(
    f"Photos stored: {total_photos}"
)

print(
    f"Faces stored:  {total_faces}"
)

print("=" * 60)


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()

print("Database connection closed.")