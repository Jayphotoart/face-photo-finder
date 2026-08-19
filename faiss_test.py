import cv2
import numpy as np
import faiss

from insightface.app import FaceAnalysis


# -----------------------------------
# 1. InsightFace initialize
# -----------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    root=r"G:\AI Face Photo Finder\insightface_models",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# -----------------------------------
# 2. Face embedding function
# -----------------------------------

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR:", image_path, "not found")
        return None

    faces = app.get(image)

    if len(faces) == 0:
        print("No face found:", image_path)
        return None

    # Largest face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                       (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding.astype("float32")

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# -----------------------------------
# 3. Create embeddings
# -----------------------------------

embedding1 = get_embedding("images/test.jpg")
embedding2 = get_embedding("images/test2.jpg")

if embedding1 is None or embedding2 is None:
    exit()


# -----------------------------------
# 4. Create FAISS index
# -----------------------------------

dimension = 512

index = faiss.IndexFlatIP(dimension)


# -----------------------------------
# 5. Add embeddings
# -----------------------------------

database_vectors = np.vstack([
    embedding1,
    embedding2
]).astype("float32")

index.add(database_vectors)

print()
print("FAISS database created")
print("Total vectors:", index.ntotal)


# -----------------------------------
# 6. Search
# -----------------------------------

query = embedding1.reshape(1, -1).astype("float32")

k = 2

scores, ids = index.search(query, k)


# -----------------------------------
# 7. Results
# -----------------------------------

print()
print("Search Results")
print("----------------")

for rank, (score, face_id) in enumerate(
    zip(scores[0], ids[0]),
    start=1
):

    print(
        f"Rank {rank}: "
        f"Face ID = {face_id}, "
        f"Similarity = {score:.4f}"
    )