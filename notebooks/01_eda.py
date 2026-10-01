import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH, TRAIN_PATH, TEST_PATH, EMOTIONS, verify_dataset


def run_eda():
    """
    Performs Exploratory Data Analysis (EDA) on the FER-2013 dataset.
    """
    print("=" * 60)
    print("FER-2013 DATASET VERIFICATION & EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset verification as required
    train_counts, test_counts = verify_dataset()

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    grand_total = total_train + total_test

    print("\n" + "=" * 60)
    print("CLASS PROPORTIONS & IMBALANCE ANALYSIS")
    print("=" * 60)
    print(f"{'Class':<12} {'Train Count':<12} {'Train %':<10} {'Test Count':<12} {'Test %':<10}")
    print("-" * 58)
    for c in EMOTIONS:
        tr_cnt = train_counts[c]
        te_cnt = test_counts[c]
        tr_pct = (tr_cnt / total_train) * 100
        te_pct = (te_cnt / total_test) * 100
        print(f"{c:<12} {tr_cnt:<12} {tr_pct:<9.2f}% {te_cnt:<12} {te_pct:<9.2f}%")

    print("\nKey Observation:")
    disgust_pct = (train_counts['disgust'] / total_train) * 100
    happy_pct = (train_counts['happy'] / total_train) * 100
    print(f"- 'disgust' is the minority class ({train_counts['disgust']} samples, {disgust_pct:.2f}% of train).")
    print(f"- 'happy' is the majority class ({train_counts['happy']} samples, {happy_pct:.2f}% of train).")
    print(f"- Ratio between majority and minority: {train_counts['happy'] / train_counts['disgust']:.1f}x.")
    print("  -> Class-weighted CrossEntropyLoss is critical to prevent ignoring disgust.")

    # 2. Image dimension & pixel intensity verification
    print("\n" + "=" * 60)
    print("IMAGE INTEGRITY & PIXEL RANGE VERIFICATION")
    print("=" * 60)

    shapes = set()
    sample_pixels = []
    checked_count = 0

    # Sample images from each class
    for c in EMOTIONS:
        class_files = list((TRAIN_PATH / c).glob("*.*"))[:100]
        for f in class_files:
            img = Image.open(f)
            arr = np.array(img)
            shapes.add(arr.shape)
            sample_pixels.append(arr)
            checked_count += 1

    sample_arr = np.stack(sample_pixels)
    print(f"Checked images sample size: {checked_count}")
    print(f"Observed shapes: {shapes} (Expected: (48, 48))")
    print(f"Pixel min: {sample_arr.min()}, max: {sample_arr.max()}")
    print(f"Mean pixel intensity: {sample_arr.mean():.2f}")
    print(f"Standard deviation:   {sample_arr.std():.2f}")
    print("=" * 60)
    print("[SUCCESS] FER-2013 EDA & verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
