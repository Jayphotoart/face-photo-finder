import cv2

# Photo load કરો
image = cv2.imread("images/test.jpg")

if image is None:
    print("ERROR: Photo મળી નથી.")
    exit()

print("Photo successfully loaded!")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Faces detect કરો
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

print("Faces found:", len(faces))

# દરેક face પર rectangle
for i, (x, y, w, h) in enumerate(faces, start=1):
    print(f"Face {i}: x={x}, y={y}, width={w}, height={h}")

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

# Result save કરો
output_path = "images/result.jpg"
cv2.imwrite(output_path, image)

print("Result saved:", output_path)