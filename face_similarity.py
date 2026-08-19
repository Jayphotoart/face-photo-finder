import cv2
import numpy as np
from insightface.app import FaceAnalysis

# InsightFace initialize
app = FaceAnalysis(
    name="buffalo_l",
    root=r"G:\AI Face Photo Finder\insightface_models",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: {image_path} મળી નથી.")
        return None

    faces = app.get(image)

    if len(faces) == 0:
        print(f"ERROR: {image_path} માં face મળ્યો નથી.")
        return None

    # સૌથી મોટો face પસંદ કરીએ
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                       (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# બંને photosના embeddings
embedding1 = get_embedding("images/test.jpg")
embedding2 = get_embedding("images/test2.jpg")

if embedding1 is None or embedding2 is None:
    exit()

# Cosine similarity
similarity = np.dot(embedding1, embedding2)

print()
print("Face Similarity Result")
print("----------------------")
print("Photo 1 embedding:", embedding1.shape)
print("Photo 2 embedding:", embedding2.shape)
print("Cosine similarity:", similarity)