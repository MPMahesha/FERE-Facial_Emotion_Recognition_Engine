"""
Text Emotion Recognition - Training Script
==========================================

Trains a beginner-friendly, fully local NLP classifier that maps a short text
sentence to one of six emotions:

    sadness, joy, love, anger, fear, surprise

Pipeline (kept deliberately simple so it is easy to explain in the viva):

    1. Input text
    2. Normalisation (lowercase + whitespace cleanup)
    3. TF-IDF feature extraction (scikit-learn)
    4. Logistic Regression classifier
    5. Output: predicted emotion, confidence, per-class probabilities

The whole thing is wrapped in a single scikit-learn Pipeline so that the exact
same preprocessing is applied at training time and at inference time.

Dataset
-------
DAIR.AI Emotion dataset from Hugging Face, loaded with the predefined splits:

    dataset = load_dataset("dair-ai/emotion", "split")

We train on the "train" split, use the "validation" split only to pick the
regularisation strength C, then fit the final model on train + validation and
report metrics once on the untouched "test" split.

Run
---
    python -m nlp.train_text_model

Artifacts are written to models/text_emotion/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from nlp.text_preprocessing import normalize_text

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "text_emotion"
PIPELINE_PATH = MODELS_DIR / "text_emotion_pipeline.joblib"
LABELS_PATH = MODELS_DIR / "label_mapping.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

RANDOM_SEED = 42


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------
def load_emotion_dataset():
    """Load the DAIR.AI emotion dataset with its predefined splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - guidance for the user
        raise SystemExit(
            "The 'datasets' package is required. Install it with:\n"
            "    pip install datasets"
        ) from exc

    print("Loading dataset 'dair-ai/emotion' (config: split)...")
    try:
        dataset = load_dataset("dair-ai/emotion", "split")
    except Exception as exc:  # network / cache problems
        raise SystemExit(
            "Could not download the DAIR.AI emotion dataset.\n"
            "The first run needs internet access to https://huggingface.co.\n"
            "Check your connection, or pre-download the dataset and re-run.\n"
            f"Original error: {exc}"
        ) from exc

    return dataset


# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
def build_pipeline(c_value: float):
    """Return the TF-IDF + Logistic Regression pipeline with a fixed seed.

    Features combine word n-grams (1-2) with character n-grams (2-5). Character
    features help with informal/short text and misspellings; word features carry
    most of the emotion signal. English stop words are removed from the word
    features (common words like "you/me/i" otherwise blur the joy vs love
    boundary), while the character features keep the full string.
    ``class_weight='balanced'`` counteracts the dataset's skew towards
    joy/sadness so the smaller classes (love, surprise) are not ignored.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    word_features = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    char_features = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )

    return Pipeline(
        steps=[
            ("features", FeatureUnion([("word", word_features), ("char", char_features)])),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# ------------------------------------------------------------------
# Training + evaluation
# ------------------------------------------------------------------
def train() -> None:
    np.random.seed(RANDOM_SEED)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    dataset = load_emotion_dataset()
    label_names = dataset["train"].features["label"].names
    print(f"Classes ({len(label_names)}): {label_names}")

    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    val_texts = list(dataset["validation"]["text"])
    val_labels = list(dataset["validation"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    print(
        f"Split sizes -> train: {len(train_texts)}, "
        f"validation: {len(val_texts)}, test: {len(test_texts)}"
    )

    # ---- Model selection on the validation split ----
    candidate_cs = [1.0, 1.5, 2.0, 3.0]
    best_c = candidate_cs[0]
    best_val_f1 = -1.0

    print("\nSelecting regularisation strength C on the validation split:")
    for c in candidate_cs:
        pipeline = build_pipeline(c)
        pipeline.fit(train_texts, train_labels)
        val_pred = pipeline.predict(val_texts)
        val_f1 = f1_score(val_labels, val_pred, average="macro")
        print(f"  C={c:<4} validation macro-F1 = {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c

    print(f"Selected C = {best_c} (validation macro-F1 = {best_val_f1:.4f})")

    # ---- Final fit on train + validation, single evaluation on test ----
    final_pipeline = build_pipeline(best_c)
    final_pipeline.fit(train_texts + val_texts, train_labels + val_labels)

    test_pred = final_pipeline.predict(test_texts)

    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred, average="macro", zero_division=0)
    recall = recall_score(test_labels, test_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(test_labels, test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_labels, test_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(test_labels, test_pred, labels=list(range(len(label_names))))

    print("\n===== Final test-set evaluation =====")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall   : {recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(test_labels, test_pred, target_names=label_names, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)

    # ---- Persist artifacts ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, PIPELINE_PATH)

    label_mapping = {
        "labels": label_names,
        "label_to_index": {name: i for i, name in enumerate(label_names)},
        "index_to_label": {str(i): name for i, name in enumerate(label_names)},
        "model_type": "TF-IDF (word 1-2 stopwords + char_wb 2-5) + LogisticRegression (balanced)",
        "dataset": "dair-ai/emotion (config: split)",
        "selected_C": best_c,
        "validation_macro_f1": round(float(best_val_f1), 4),
    }
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        json.dump(label_mapping, fh, indent=2)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": label_names,
        "test_size": len(test_labels),
        "train_size": len(train_texts + val_texts),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    save_confusion_matrix(cm, label_names)
    save_model_card(accuracy, macro_f1, best_c, label_names)

    print(f"\nSaved pipeline      -> {PIPELINE_PATH}")
    print(f"Saved label mapping -> {LABELS_PATH}")
    print(f"Saved metrics       -> {METRICS_PATH}")
    print(f"Saved confusion mat -> {CONFUSION_MATRIX_PATH}")


def save_confusion_matrix(cm, label_names) -> None:
    """Render a readable confusion-matrix heatmap next to the other model artifacts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping confusion matrix image.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title("Text Emotion Classifier - Test Confusion Matrix")

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def save_model_card(accuracy, macro_f1, best_c, label_names) -> None:
    """Human-readable summary for the README / viva."""
    card = {
        "pipeline": "FeatureUnion[word TF-IDF (1-2, stopwords) , char_wb TF-IDF (2-5)] -> LogisticRegression (class_weight=balanced)",
        "random_seed": RANDOM_SEED,
        "selected_C": best_c,
        "labels": label_names,
        "test_accuracy": round(float(accuracy), 4),
        "test_macro_f1": round(float(macro_f1), 4),
    }
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        sys.exit("Training interrupted by user.")
