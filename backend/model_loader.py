import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ============================================================
# PYTORCH CNN ARCHITECTURE (READY FOR MEMBER A)
# ============================================================

class EmotionCNN(nn.Module):
    """
    Standard PyTorch CNN designed for 48x48 Grayscale Facial Emotion Recognition (FER-2013).
    Member A can use this standard architecture or load compatible weights.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # Dense Classifier
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


# ============================================================
# MODEL LOADER & INFERENCE
# ============================================================

class EmotionModelManager:
    """
    Manages loading the trained PyTorch model from the models/ directory.
    Falls back gracefully to a mock prediction engine if no .pth model exists yet.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_mock = True
        self.model_filename = None
        self.load_model()

    def load_model(self):
        """
        Attempts to locate and load a .pth PyTorch model from models_dir.
        """
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)

        pth_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pth") or f.endswith(".pt")]

        if pth_files:
            target_model = os.path.join(self.models_dir, pth_files[0])
            try:
                # Initialize architecture and load weights
                model = EmotionCNN(num_classes=len(EMOTIONS))
                state_dict = torch.load(target_model, map_location=self.device)
                
                # Support both direct state_dict and full checkpoint dicts
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    model.load_state_dict(state_dict["state_dict"])
                elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                elif isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                else:
                    model = state_dict

                model.to(self.device)
                model.eval()
                self.model = model
                self.is_mock = False
                self.model_filename = pth_files[0]
                print(f"[INFO] Loaded trained PyTorch model: {pth_files[0]}")
            except Exception as e:
                print(f"[WARN] Failed to load {pth_files[0]}: {e}. Falling back to mock engine.")
                self.model = None
                self.is_mock = True
        else:
            print("[INFO] No .pth model found in models/. Using temporary mock prediction engine.")
            self.model = None
            self.is_mock = True

    def _mock_prediction(self) -> tuple[str, float, dict[str, float]]:
        """
        Generates realistic mock emotion predictions for frontend development and testing.
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

        # Fix minor rounding discrepancies
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
            # Preprocess: convert to 48x48 grayscale image
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize pixel intensities between 0.0 and 1.0
            normalized = resized.astype("float32") / 255.0

            # Convert to PyTorch Tensor format: [Batch=1, Channel=1, Height=48, Width=48]
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

            probabilities = {
                EMOTIONS[i]: round(float(probs[i]), 4)
                for i in range(len(EMOTIONS))
            }

            best_idx = int(np.argmax(probs))
            primary_emotion = EMOTIONS[best_idx]
            confidence = round(float(probs[best_idx]), 4)

            return primary_emotion, confidence, probabilities

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}. Returning mock fallback.")
            return self._mock_prediction()


# Global Model Manager Instance
model_manager = EmotionModelManager()
