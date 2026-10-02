import base64
import cv2
import numpy as np


# ============================================================
# FACE DETECTION INITIALIZATION
# ============================================================

# Load OpenCV's pre-trained Haar Cascade for frontal face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(image_input: str) -> np.ndarray:
    """
    Decodes a base64 string or Data URL into an OpenCV BGR image array.
    """
    if not image_input or not isinstance(image_input, str):
        raise ValueError("Image input must be a non-empty string.")

    # Remove data URL header if present (e.g. 'data:image/jpeg;base64,...')
    if "," in image_input:
        image_input = image_input.split(",", 1)[1]

    # Decode base64 bytes to image array
    image_bytes = base64.b64decode(image_input)
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image data into an OpenCV array.")

    return image


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    """
    Detects faces in an image using OpenCV Haar Cascade.
    Returns:
        - boxes: list of bounding box dicts with x, y, width, height
        - face_crops: list of cropped face sub-images (BGR)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect faces with standard scaling and neighbor parameters
    detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    boxes = []
    face_crops = []

    for (x, y, w, h) in detected:
        boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
        # Crop the detected face region
        cropped = image[y:y + h, x:x + w]
        face_crops.append(cropped)

    # Fallback for pre-cropped face images (e.g. FER-2013 benchmark samples <= 128x128)
    if not boxes:
        h, w = image.shape[:2]
        if h <= 128 and w <= 128:
            boxes.append({"x": 0, "y": 0, "width": int(w), "height": int(h)})
            face_crops.append(image)

    return boxes, face_crops

