import cv2

IMAGE_PATH = "test_image.jpg"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image could not be loaded")
    exit()

# Get image dimensions
height, width = image.shape[:2]

print("Image size:", width, "x", height)

# Load YuNet
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (width, height),
    0.6,
    0.3,
    5000
)

print("✅ YuNet model loaded")

# Detect faces
_, faces = detector.detect(image)

if faces is None or len(faces) == 0:

    print("❌ No human face detected")

else:

    print("✅ Face detection successful")
    print("Faces detected:", len(faces))

    for i, face in enumerate(faces):

        x, y, w, h = face[:4].astype(int)

        print(
            f"Face {i + 1}: "
            f"x={x}, y={y}, width={w}, height={h}"
        )