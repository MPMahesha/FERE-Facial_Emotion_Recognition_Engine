import os
import random
from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ============================================================
# CUSTOM VGG-STYLE CNN ARCHITECTURE
# MUST BE IDENTICAL TO models/emotion_cnn.py
# ============================================================

class EmotionCNN(nn.Module):
    """
    Standard VGG-style Deep Convolutional Neural Network designed from scratch
    for 48x48 Grayscale Facial Emotion Recognition (FER-2013).

    Architecture:
        - Block 1 (48x48 -> 24x24):
            Conv2d(1 -> 32, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(32 -> 64, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Block 2 (24x24 -> 12x12):
            Conv2d(64 -> 128, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(128 -> 128, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Block 3 (12x12 -> 6x6):
            Conv2d(128 -> 256, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(256 -> 256, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Classifier:
            AdaptiveAvgPool2d((3, 3)) -> Flatten (2304)
            Linear(256 * 3 * 3 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.5)
            Linear(512 -> num_classes)
    """
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()

        # ConvBlock 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        # ConvBlock 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # ConvBlock 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        # Dense Classifier
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvBlock 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))

        # ConvBlock 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))

        # ConvBlock 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))

        # Dense Classification
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts intermediate feature activations from each convolutional block.
        """
        b1 = self.relu(self.bn1_1(self.conv1_1(x)))
        b1 = self.relu(self.bn1_2(self.conv1_2(b1)))

        b2 = self.drop1(self.pool1(b1))
        b2 = self.relu(self.bn2_1(self.conv2_1(b2)))
        b2 = self.relu(self.bn2_2(self.conv2_2(b2)))

        b3 = self.drop2(self.pool2(b2))
        b3 = self.relu(self.bn3_1(self.conv3_1(b3)))
        b3 = self.relu(self.bn3_2(self.conv3_2(b3)))

        return {
            "block1": b1,
            "block2": b2,
            "block3": b3
        }


# ============================================================
# PREDICTION RESULT CONTAINER
# ============================================================

class EmotionPredictionResult(tuple):
    """
    Result container that behaves as:
    1. A 3-element tuple: (emotion, confidence, probabilities)
    2. An object with attributes: .emotion, .confidence, .probabilities
    3. A mapping: ['emotion'], ['confidence'], ['probabilities']
    """
    def __new__(cls, emotion: str, confidence: float, probabilities: dict[str, float]):
        return super().__new__(cls, (emotion, confidence, probabilities))

    @property
    def emotion(self) -> str:
        return self[0]

    @property
    def confidence(self) -> float:
        return self[1]

    @property
    def probabilities(self) -> dict[str, float]:
        return self[2]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "emotion":
                return self[0]
            elif item == "confidence":
                return self[1]
            elif item == "probabilities":
                return self[2]
            raise KeyError(f"Invalid key '{item}'. Allowed: 'emotion', 'confidence', 'probabilities'")
        return super().__getitem__(item)

    def get(self, key, default=None):
        if key in ("emotion", "confidence", "probabilities"):
            return self[key]
        return default

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities
        }

    def __repr__(self) -> str:
        return f"EmotionPrediction(emotion='{self.emotion}', confidence={self.confidence}, probabilities={self.probabilities})"


# ============================================================
# PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================

def preprocess_image(image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Preprocesses an input image into a standardized 48x48 single-channel normalized tensor:
    Shape: [1, 1, 48, 48], dtype: float32, range: [0.0, 1.0].
    """
    if isinstance(image, (str, Path)):
        img_path = str(image)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if cv_img is None:
            raise ValueError(f"Failed to read image at: {img_path}")
        gray = cv_img
    elif isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                gray = image.squeeze(axis=2)
            else:
                gray = image[:, :, 0]
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    elif isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.max() > 1.0:
            t = t / 255.0
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] == 1:
                t = t.unsqueeze(0)
            elif t.shape[2] == 1:
                t = t.permute(2, 0, 1).unsqueeze(0)
            elif t.shape[0] in (3, 4):
                # RGB/RGBA tensor to grayscale
                t = (0.2989 * t[0] + 0.5870 * t[1] + 0.1140 * t[2]).unsqueeze(0).unsqueeze(0)
        if t.shape[-2:] != (48, 48):
            t = F.interpolate(t, size=(48, 48), mode="bilinear", align_corners=False)
        return t
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure 48x48 resolution
    if gray.shape != (48, 48):
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize [0.0, 1.0] float32
    norm = gray.astype(np.float32)
    if norm.max() > 1.0:
        norm = norm / 255.0

    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    return tensor


def load_model(model_path: str = None, device: torch.device = None) -> EmotionCNN:
    """
    Loads the trained EmotionCNN PyTorch model from models/emotion_cnn.pth.
    Validates state-dict with strict=True to guarantee 100% parameter match.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        backend_dir = Path(__file__).resolve().parent
        candidates = [
            backend_dir.parent / "models" / "emotion_cnn.pth",
            Path("models/emotion_cnn.pth").resolve(),
            Path("models/emotion_cnn.pth"),
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break
        if model_path is None:
            model_path = str(candidates[0])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = EmotionCNN(num_classes=len(EMOTIONS))
    state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # strict=True verifies there are no missing or unexpected keys
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_emotion(
    model: nn.Module,
    image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
    device: torch.device = None
) -> EmotionPredictionResult:
    """
    Runs emotion prediction on an image using the trained EmotionCNN model.
    Returns:
        EmotionPredictionResult with:
        - emotion: dominant emotion string ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        - confidence: confidence score (float 0.0 - 1.0)
        - probabilities: dict mapping all 7 emotions to their predicted probabilities
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = preprocess_image(image).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        flipped_outputs = model(torch.flip(tensor, dims=[3]))
        probs = (
            (F.softmax(outputs, dim=1) + F.softmax(flipped_outputs, dim=1)) / 2
        ).squeeze(0).cpu().numpy()

    probabilities = {
        EMOTIONS[i]: round(float(probs[i]), 4)
        for i in range(len(EMOTIONS))
    }

    best_idx = int(np.argmax(probs))
    emotion = EMOTIONS[best_idx]
    confidence = round(float(probs[best_idx]), 4)

    return EmotionPredictionResult(emotion=emotion, confidence=confidence, probabilities=probabilities)


# ============================================================
# MODEL MANAGER (BACKWARD COMPATIBLE WITH FASTAPI & MOCK FALLBACK)
# ============================================================

class EmotionModelManager:
    """
    Manages loading the trained PyTorch model and executing inference for FastAPI.
    Falls back gracefully to mock prediction if model file is unavailable.
    """
    def __init__(self, models_dir="models"):
        backend_dir = Path(__file__).resolve().parent
        self.models_dir = str(backend_dir.parent / models_dir) if not os.path.isabs(models_dir) else models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_mock = True
        self.model_filename = None
        self.load_model()

    def load_model(self):
        """
        Attempts to locate and load emotion_cnn.pth from models_dir.
        """
        target_path = os.path.join(self.models_dir, "emotion_cnn.pth")
        if not os.path.exists(target_path):
            pth_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pth") or f.endswith(".pt")] if os.path.exists(self.models_dir) else []
            if pth_files:
                target_path = os.path.join(self.models_dir, pth_files[0])
            else:
                target_path = None

        if target_path and os.path.exists(target_path):
            try:
                self.model = load_model(target_path, device=self.device)
                self.is_mock = False
                self.model_filename = os.path.basename(target_path)
                print(f"[INFO] Loaded trained PyTorch model: {self.model_filename} on {self.device}")
            except Exception as e:
                print(f"[WARN] Failed to load {target_path}: {e}. Falling back to mock engine.")
                self.model = None
                self.is_mock = True
        else:
            print("[INFO] No .pth model found in models/. Using temporary mock prediction engine.")
            self.model = None
            self.is_mock = True

    def _mock_prediction(self) -> tuple[str, float, dict[str, float]]:
        """
        Fallback mock prediction if model is not loaded.
        """
        primary_emotion = random.choice(EMOTIONS)
        primary_confidence = round(random.uniform(0.70, 0.95), 4)

        remaining_budget = round(1.0 - primary_confidence, 4)
        other_emotions = [e for e in EMOTIONS if e != primary_emotion]
        random_splits = [random.random() for _ in other_emotions]
        total_split = sum(random_splits)

        probabilities = {}
        for emotion, split in zip(other_emotions, random_splits):
            prob = round((split / total_split) * remaining_budget, 4)
            probabilities[emotion] = prob

        probabilities[primary_emotion] = primary_confidence
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: round(v / total_prob, 4) for k, v in probabilities.items()}

        return primary_emotion, probabilities[primary_emotion], probabilities

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Runs emotion prediction on a cropped face image (BGR).
        Returns:
            - emotion: dominant emotion label (e.g. 'happy')
            - confidence: confidence score (float 0.0 - 1.0)
            - probabilities: dictionary mapping all 7 emotions to probabilities
        """
        if self.is_mock or self.model is None:
            return self._mock_prediction()

        try:
            return predict_emotion(self.model, face_bgr, device=self.device)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}. Falling back to mock prediction.")
            return self._mock_prediction()


# Global Model Manager Instance
model_manager = EmotionModelManager()
