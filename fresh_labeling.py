import os
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = r"G:\AI Face Photo Finder\images"
OUTPUT_FILE = r"G:\AI Face Photo Finder\fresh_labels.json"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("FRESH FACE LABELING")
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
# IMAGE LIST
# ============================================================

extensions = (".jpg", ".jpeg", ".png")

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(extensions)
])

print(f"\nImages found: {len(files)}")

if not files:
    print("ERROR: No images found.")
    raise SystemExit


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

records = []

if os.path.exists(OUTPUT_FILE):

    try:
        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            records = json.load(f)

        print(
            f"Existing labeling loaded: "
            f"{len(records)} faces"
        )

    except Exception:
        print(
            "WARNING: Could not read existing "
            "fresh_labels.json"
        )

        records = []


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_records():

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=2
        )


# ============================================================
# FIND EXISTING RECORD
# ============================================================

def find_record(filename, face_index):

    for record in records:

        if (
            record["filename"] == filename
            and
            record["face_index"] == face_index
        ):
            return record

    return None


# ============================================================
# LABEL INPUT
# ============================================================

def get_label():

    print()
    print("--------------------------------------------------")
    print("LABEL OPTIONS")
    print("--------------------------------------------------")
    print("A = Person A")
    print("B = Person B")
    print("C = Person C")
    print("D = Person D")
    print("S = SKIP")
    print("R = RESTART / change previous label")
    print("Q = QUIT and save progress")
    print("--------------------------------------------------")

    while True:

        value = input(
            "Enter label [A/B/C/D/S/R/Q]: "
        ).strip().upper()

        if value in (
            "A",
            "B",
            "C",
            "D",
            "S",
            "R",
            "Q"
        ):
            return value

        print(
            "Invalid input. "
            "Use A, B, C, D, S, R or Q."
        )


# ============================================================
# PROCESS
# ============================================================

total_faces = 0
new_labels = 0
skipped = 0


for photo_number, filename in enumerate(
    files,
    start=1
):

    print()
    print("=" * 70)
    print(
        f"PHOTO {photo_number}/{len(files)}"
    )
    print(
        f"FILE: {filename}"
    )
    print("=" * 70)

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        continue


    faces = app.get(image)

    if len(faces) == 0:

        print(
            "No faces detected."
        )

        continue


    # --------------------------------------------------------
    # LEFT -> RIGHT
    # --------------------------------------------------------

    faces = sorted(
        faces,
        key=lambda face: face.bbox[0]
    )

    print(
        f"Faces detected: {len(faces)}"
    )


    # --------------------------------------------------------
    # EACH FACE
    # --------------------------------------------------------

    face_index = 0

    while face_index < len(faces):

        face = faces[face_index]

        total_faces += 1

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        # ----------------------------------------------------
        # CREATE DISPLAY
        # ----------------------------------------------------

        display = image.copy()


        # Draw ALL faces first
        for i, other_face in enumerate(faces):

            bx = other_face.bbox.astype(int)

            ox1, oy1, ox2, oy2 = bx

            if i == face_index:

                thickness = 5

                cv2.rectangle(
                    display,
                    (ox1, oy1),
                    (ox2, oy2),
                    (0, 255, 0),
                    thickness
                )

                cv2.putText(
                    display,
                    f"CURRENT FACE {i}",
                    (
                        ox1,
                        max(40, oy1 - 15)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3
                )

            else:

                cv2.rectangle(
                    display,
                    (ox1, oy1),
                    (ox2, oy2),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Face {i}",
                    (
                        ox1,
                        max(25, oy1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )


        # ----------------------------------------------------
        # INFO PANEL
        # ----------------------------------------------------

        panel_height = 150

        panel = np.zeros(
            (
                panel_height,
                display.shape[1],
                3
            ),
            dtype=np.uint8
        )

        cv2.putText(
            panel,
            f"FILE: {filename}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            f"CURRENT FACE: {face_index}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            panel,
            "A/B/C/D = LABEL | S = SKIP | R = BACK | Q = QUIT",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        display = np.vstack(
            [panel, display]
        )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        h, w = display.shape[:2]

        max_width = 1400
        max_height = 900

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
            "FRESH LABELING"
        )

        cv2.imshow(
            window_name,
            display
        )

        cv2.waitKey(1)


        # ----------------------------------------------------
        # TERMINAL INPUT
        # ----------------------------------------------------

        print()
        print(
            f"PHOTO {photo_number}/{len(files)}"
        )

        print(
            f"FILE      : {filename}"
        )

        print(
            f"CURRENT FACE : {face_index}"
        )

        print(
            f"Position: "
            f"x={x1}, y={y1}, "
            f"x2={x2}, y2={y2}"
        )

        label = get_label()


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if label == "Q":

            cv2.destroyAllWindows()

            save_records()

            print()
            print(
                "Progress saved."
            )

            print(
                f"Saved records: {len(records)}"
            )

            print(
                "Labeling stopped."
            )

            raise SystemExit


        # ----------------------------------------------------
        # BACK / R
        # ----------------------------------------------------

        if label == "R":

            if face_index > 0:

                face_index -= 1

                # Remove previous record
                old_record = find_record(
                    filename,
                    face_index
                )

                if old_record:

                    records.remove(
                        old_record
                    )

                    save_records()

                    print(
                        f"Previous label for "
                        f"Face {face_index} removed."
                    )

                continue

            else:

                print(
                    "Already at first face."
                )

                continue


        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        if label == "S":

            existing = find_record(
                filename,
                face_index
            )

            if existing:

                records.remove(
                    existing
                )

            records.append(
                {
                    "filename": filename,
                    "face_index": int(face_index),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],
                    "person_label": "SKIP",
                    "embedding": None
                }
            )

            save_records()

            skipped += 1

            print(
                f"SKIPPED -> "
                f"{filename} | "
                f"Face={face_index}"
            )

            face_index += 1

            continue


        # ----------------------------------------------------
        # A / B / C / D
        # ----------------------------------------------------

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32
        )


        existing = find_record(
            filename,
            face_index
        )

        if existing:

            records.remove(
                existing
            )


        records.append(
            {
                "filename": filename,
                "face_index": int(face_index),
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                "person_label": label,
                "embedding": embedding.tolist()
            }
        )


        save_records()

        new_labels += 1

        print()
        print(
            f"SAVED -> "
            f"{filename} | "
            f"Face={face_index} | "
            f"Person={label}"
        )

        face_index += 1


    cv2.destroyAllWindows()


# ============================================================
# FINISHED
# ============================================================

cv2.destroyAllWindows()

save_records()

print()
print("=" * 70)
print("FRESH LABELING COMPLETED")
print("=" * 70)

print(
    f"Photos scanned : {len(files)}"
)

print(
    f"New labels     : {new_labels}"
)

print(
    f"Skipped        : {skipped}"
)

print(
    f"Total records  : {len(records)}"
)

print()
print(
    f"Output file:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)