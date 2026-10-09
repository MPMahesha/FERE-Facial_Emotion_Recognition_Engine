"""
Text Emotion Recognition - Model Loader & Inference
====================================================

Loads the trained scikit-learn pipeline (TF-IDF + Logistic Regression) that is
produced by ``nlp/train_text_model.py`` and exposes a small manager used by the
FastAPI ``/predict-text`` endpoint.

Important: these six text-emotion classes are DIFFERENT from the seven facial
FER-2013 classes. Text and face models must not be presented as identical.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Canonical order of the DAIR.AI emotion dataset labels.
TEXT_EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


class TextEmotionModelManager:
    """
    Loads the saved text-emotion pipeline once and runs inference.

    Unlike the facial model, there is NO mock fallback: if the trained model is
    missing we report that clearly instead of returning fake predictions.
    """

    def __init__(self, models_dir: str | Path = "models/text_emotion"):
        backend_dir = Path(__file__).resolve().parent
        self.models_dir = (
            Path(models_dir)
            if os.path.isabs(models_dir)
            else backend_dir.parent / models_dir
        )
        self.pipeline_path = self.models_dir / "text_emotion_pipeline.joblib"
        self.labels_path = self.models_dir / "label_mapping.json"

        self.pipeline = None
        self.is_available = False
        self.load_error = None
        self.index_to_label = {i: label for i, label in enumerate(TEXT_EMOTIONS)}
        self.labels = list(TEXT_EMOTIONS)
        self.metadata = {}
        self._loaded_mtime = None

        self.load_model()

    def _model_mtime(self):
        """Modification time of the model file, or None if it is missing."""
        try:
            return self.pipeline_path.stat().st_mtime
        except OSError:
            return None

    def maybe_reload(self) -> None:
        """
        Reload the pipeline if the file on disk changed since it was loaded.

        This means a freshly retrained model is picked up automatically on the
        next request - no server restart required, and no risk of silently
        serving stale predictions.
        """
        current = self._model_mtime()
        if current is None:
            return
        if self._loaded_mtime is None or current != self._loaded_mtime:
            print("[INFO] Text-emotion model file changed on disk - reloading.")
            self.load_model()

    def load_model(self) -> None:
        """Load the joblib pipeline and label mapping if present."""
        if not self.pipeline_path.exists():
            self.load_error = (
                "Text-emotion model not found. Train it first with:\n"
                "    python -m nlp.train_text_model"
            )
            print(f"[INFO] {self.load_error}".replace("\n", " "))
            return

        try:
            import joblib

            self.pipeline = joblib.load(self.pipeline_path)

            if self.labels_path.exists():
                with open(self.labels_path, "r", encoding="utf-8") as fh:
                    mapping = json.load(fh)
                self.labels = mapping.get("labels", list(TEXT_EMOTIONS))
                self.index_to_label = {
                    int(k): v for k, v in mapping.get("index_to_label", {}).items()
                }
                self.metadata = mapping

            self._loaded_mtime = self._model_mtime()
            self.is_available = True
            self.load_error = None
            print(f"[INFO] Loaded text-emotion pipeline: {self.pipeline_path.name}")
        except Exception as exc:  # pragma: no cover - depends on local files
            self.pipeline = None
            self.is_available = False
            self._loaded_mtime = None
            self.load_error = f"Failed to load text-emotion model: {exc}"
            print(f"[WARN] {self.load_error}")

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        """
        Predict the emotion of ``text``.

        Returns:
            (emotion, confidence, probabilities)
            - emotion: dominant label string
            - confidence: probability of the dominant label (0.0 - 1.0)
            - probabilities: dict mapping every label to its probability
        """
        # Pick up a retrained model without requiring a server restart.
        self.maybe_reload()

        if not self.is_available or self.pipeline is None:
            raise RuntimeError(self.load_error or "Text-emotion model unavailable.")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text is empty.")

        proba = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)

        probabilities = {}
        for class_index, prob in zip(classes, proba):
            label = self.index_to_label.get(int(class_index), str(class_index))
            probabilities[label] = round(float(prob), 4)

        # Make sure every known label is represented (defensive / stable UI).
        for label in self.labels:
            probabilities.setdefault(label, 0.0)

        best_label = max(probabilities, key=probabilities.get)
        confidence = round(float(probabilities[best_label]), 4)
        return best_label, confidence, probabilities


# Global manager instance - loaded once at import time, reused by every request.
text_model_manager = TextEmotionModelManager()
