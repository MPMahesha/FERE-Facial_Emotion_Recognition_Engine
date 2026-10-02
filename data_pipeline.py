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
    Supports in-memory caching for blazingly fast training throughput.
    """
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None, preload: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload
        self.cached_images = None

        if self.preload:
            # Pre-load all 48x48 images into memory (takes only ~60MB RAM)
            self.cached_images = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.cached_images is not None:
            image = Image.fromarray(self.cached_images[idx])
        else:
            image = Image.open(self.image_paths[idx]).convert("L")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


# ============================================================
# DATA TRANSFORMS & AUGMENTATION
# ============================================================

def get_transforms(augment: bool = True):
    """
    Data augmentation for training (realistic facial transformations:
    horizontal flip, small rotation, mild crop/translation, subtle lighting).
    Deterministic tensor conversion for validation and testing.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(48, padding=4, padding_mode="reflect"),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
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

def compute_class_weights(labels: list[int], num_classes: int = NUM_CLASSES, method: str = "sqrt") -> torch.Tensor:
    """
    Computes class weights to address class imbalance without pathological gradients.
    - 'sqrt': square-root damped inverse weights (w_c = sqrt(N_median / N_c)), normalized so mean is 1.0.
              Keeps disgust weight bounded (~2.4-3.0) rather than extreme (>9.0).
    - 'raw': raw inverse frequency (total / (C * count_c)).
    - 'none': uniform weights (all 1.0).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if method == "sqrt":
        median_count = np.median(counts)
        weights = np.sqrt(median_count / np.maximum(counts, 1.0))
        weights = weights / np.mean(weights)
    elif method == "raw":
        total_samples = len(labels)
        weights = total_samples / (num_classes * np.maximum(counts, 1.0))
    elif method == "none":
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        weights = np.ones(num_classes, dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# DATALOADERS CREATOR
# ============================================================

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
    augment: bool = True,
    weight_method: str = "sqrt"
):
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = load_split_paths(validation_split, random_seed)
    train_tf, eval_tf = get_transforms(augment=augment)

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

    class_weights = compute_class_weights(train_y, NUM_CLASSES, method=weight_method)

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
