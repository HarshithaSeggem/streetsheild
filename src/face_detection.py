import cv2
import os
import sys

# Get image path from command line
if len(sys.argv) < 2:
    print("❌ Please provide an image path.")
    print("Example: python src/face_detection.py dataset/my_test.jpg")
    exit()

image_path = sys.argv[1]

# Load image
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found:", image_path)
    exit()

print("✅ Image loaded successfully")

# Load YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

# Set image size
height, width = image.shape[:2]
face_detector.setInputSize((width, height))

# Detect faces
_, faces = face_detector.detect(image)

if faces is None:
    print("Number of faces detected: 0")
    print("❌ No face detected.")
    exit()

print("Number of faces detected:", len(faces))

# Create output folder
output_folder = "dataset/cropped_faces"
os.makedirs(output_folder, exist_ok=True)

# Crop faces
for i, face_data in enumerate(faces):

    x, y, w, h = face_data[:4]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    face = image[y:y+h, x:x+w]

    output_path = f"{output_folder}/face_{i+1}.jpg"

    cv2.imwrite(output_path, face)

    print(f"✅ Face {i+1} saved: {output_path}")

print("🎉 Face detection completed!")