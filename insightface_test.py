import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    root=r"G:\AI Face Photo Finder\insightface_models",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

# Image load
image = cv2.imread("images/test.jpg")

if image is None:
    print("ERROR: Photo મળી નથી.")
    exit()

print("Photo successfully loaded!")

# Face detection + analysis
faces = app.get(image)

print("Faces found:", len(faces))

# દરેક faceની information
for i, face in enumerate(faces, start=1):

    print(f"\nFace {i}")

    print("Bounding box:", face.bbox)

    print("Detection score:", face.det_score)

    print("Embedding shape:", face.embedding.shape)

    print("First 5 embedding values:")
    print(face.embedding[:5])