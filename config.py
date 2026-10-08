import os
from pathlib import Path

# ============================================================
# PROJECT PATHS & FER-2013 CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Centralized dataset path as requested
DATASET_PATH = Path(os.environ.get("FER_DATASET_PATH", r"Dataset\archive"))
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

# Official 7 emotion classes in canonical order
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Training & Image hyperparameters
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.15

# Saved model paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "emotion_cnn.pth"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def verify_dataset():
    """
    Verifies that the dataset path and subfolders exist and prints counts.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train path does not exist: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test path does not exist: {TEST_PATH}")

    train_classes = sorted([d.name for d in TRAIN_PATH.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in TEST_PATH.iterdir() if d.is_dir()])

    train_counts = {c: len(list((TRAIN_PATH / c).glob("*.*"))) for c in train_classes}
    test_counts = {c: len(list((TEST_PATH / c).glob("*.*"))) for c in test_classes}

    print("Dataset path:", DATASET_PATH)
    print("Detected train path:", TRAIN_PATH)
    print("Detected test path:", TEST_PATH)
    print("Number of training images:", sum(train_counts.values()))
    print("Number of test images:", sum(test_counts.values()))
    print("Classes:", train_classes)
    print("Images per class:")
    print("  Train:")
    for c in train_classes:
        print(f"    {c}: {train_counts[c]}")
    print("  Test:")
    for c in test_classes:
        print(f"    {c}: {test_counts[c]}")

    return train_counts, test_counts


if __name__ == "__main__":
    verify_dataset()
