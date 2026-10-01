import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import (
    TRAIN_PATH,
    TEST_PATH,
    EMOTIONS,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)


# ============================================================
# PYTORCH DATASET FOR FER-2013
# ============================================================

class FER2013Dataset(Dataset):
    """
    Dataset loader for 48x48 Grayscale Facial Emotion images.
    """
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load as grayscale PIL image (48x48)
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


# ============================================================
# DATA TRANSFORMS & AUGMENTATION
# ============================================================

def get_transforms():
    """
    Data augmentation for training (horizontal flip and small rotation).
    Deterministic tensor conversion for validation and testing.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


# ============================================================
# DATASET SPLITTING (STRATIFIED TRAIN / VALIDATION)
# ============================================================

def load_split_paths(validation_split: float = VALIDATION_SPLIT, random_seed: int = RANDOM_SEED):
    """
    Loads all training files and splits into train and validation sets,
    preserving class ratios (stratified split). Loads test files separately.
    """
    rng = random.Random(random_seed)

    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []

    # Process training directory by class to perform stratified split
    for label_idx, emotion in enumerate(EMOTIONS):
        class_folder = TRAIN_PATH / emotion
        files = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
        rng.shuffle(files)

        split_idx = int(len(files) * (1 - validation_split))
        class_train_files = files[:split_idx]
        class_val_files = files[split_idx:]

        train_paths.extend(class_train_files)
        train_labels.extend([label_idx] * len(class_train_files))

        val_paths.extend(class_val_files)
        val_labels.extend([label_idx] * len(class_val_files))

    # Process test directory
    test_paths = []
    test_labels = []
    for label_idx, emotion in enumerate(EMOTIONS):
        class_folder = TEST_PATH / emotion
        files = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")))
        test_paths.extend(files)
        test_labels.extend([label_idx] * len(files))

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ============================================================
# CLASS WEIGHT COMPUTATION (FOR DISGUST & IMBALANCE)
# ============================================================

def compute_class_weights(labels: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Computes inverse frequency class weights:
    weight_c = total_samples / (num_classes * class_count_c)
    Helps address class imbalance, particularly for the 'disgust' class.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# DATALOADERS CREATOR
# ============================================================

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED
):
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = load_split_paths(validation_split, random_seed)
    train_tf, eval_tf = get_transforms()

    train_dataset = FER2013Dataset(train_p, train_y, transform=train_tf)
    val_dataset = FER2013Dataset(val_p, val_y, transform=eval_tf)
    test_dataset = FER2013Dataset(test_p, test_y, transform=eval_tf)

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    class_weights = compute_class_weights(train_y, NUM_CLASSES)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "class_weights": class_weights,
        "train_labels": train_y,
        "val_labels": val_y,
        "test_labels": test_y
    }


# ============================================================
# SELF-TEST & VERIFICATION
# ============================================================

if __name__ == "__main__":
    print("[INFO] Testing FER-2013 Data Pipeline...")
    loaders = get_dataloaders(batch_size=64, num_workers=0)

    print(f"Train samples: {len(loaders['train_dataset'])}")
    print(f"Val samples:   {len(loaders['val_dataset'])}")
    print(f"Test samples:  {len(loaders['test_dataset'])}")
    print("\nClass weights (addressed for imbalance):")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:10s}: {loaders['class_weights'][i].item():.4f}")

    # Inspect first batch
    for images, labels in loaders["train_loader"]:
        print(f"\nBatch shape: images={images.shape}, labels={labels.shape}")
        print(f"Pixel range: min={images.min().item():.3f}, max={images.max().item():.3f}")
        break

    print("[SUCCESS] Data pipeline verified.")
