from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model_loader import EMOTIONS, model_manager
from backend.utils import decode_image, detect_faces


# ============================================================
# FASTAPI APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Real-Time Facial Emotion Recogniser backend using OpenCV and PyTorch.",
    version="1.0.0"
)

# Enable CORS for frontend browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image string or Data URL")


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FacePrediction(BaseModel):
    emotion: str
    confidence: float
    box: BoundingBox
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    success: bool
    face_count: int
    faces: list[FacePrediction]
    message: str | None = None


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    """
    Root health check and metadata.
    """
    return {
        "status": "online",
        "service": "Facial Emotion Recognition API",
        "version": "1.0.0",
        "emotions": EMOTIONS,
        "model_mode": "mock" if model_manager.is_mock else "trained_pytorch",
        "model_file": model_manager.model_filename
    }


@app.get("/status")
def get_status():
    """
    Returns backend readiness and model details.
    """
    return {
        "status": "ready",
        "device": str(model_manager.device),
        "is_mock": model_manager.is_mock,
        "model_file": model_manager.model_filename,
        "supported_emotions": EMOTIONS
    }


@app.post("/predict", response_model=PredictResponse)
def predict_emotion_endpoint(payload: PredictRequest):
    """
    Receives an image frame, detects faces via OpenCV, and predicts emotions using PyTorch/Mock.
    """
    # 1. Decode base64 image
    try:
        image = decode_image(payload.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # 2. Detect face regions using OpenCV Haar Cascade
    try:
        boxes, face_crops = detect_faces(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

    # 3. Handle case where no face is present
    if not boxes or not face_crops:
        return PredictResponse(
            success=True,
            face_count=0,
            faces=[],
            message="No face detected in the current frame."
        )

    # 4. Predict emotions for each detected face
    faces_results = []
    for box, crop in zip(boxes, face_crops):
        emotion, confidence, probabilities = model_manager.predict(crop)
        faces_results.append(
            FacePrediction(
                emotion=emotion,
                confidence=confidence,
                box=BoundingBox(**box),
                probabilities=probabilities
            )
        )

    return PredictResponse(
        success=True,
        face_count=len(faces_results),
        faces=faces_results,
        message="Face(s) detected and emotion(s) predicted successfully."
    )


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
