import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    EMOTIONS,
    NUM_CLASSES,
    RANDOM_SEED,
    BATCH_SIZE
)
from models.emotion_cnn import EmotionCNN
from data_pipeline import get_dataloaders


def set_seed(seed: int = RANDOM_SEED):
    """
    Sets deterministic seed across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(epochs: int = 15, batch_size: int = BATCH_SIZE, learning_rate: float = 1e-3):
    """
    Trains the custom VGG-style CNN on FER-2013 with class-weighted CrossEntropyLoss.
    Model selection is performed strictly on the validation set.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"TRAINING CUSTOM VGG-STYLE CNN ON FER-2013 (DEVICE: {device})")
    print("=" * 65)

    # 1. Load data pipeline
    pipeline = get_dataloaders(batch_size=batch_size, random_seed=RANDOM_SEED)
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    class_weights = pipeline["class_weights"].to(device)

    print(f"Training samples:   {len(pipeline['train_dataset'])}")
    print(f"Validation samples: {len(pipeline['val_dataset'])}")
    print(f"Batch size:         {batch_size}")
    print(f"Epochs:             {epochs}")
    print("\nClass weights applied in CrossEntropyLoss:")
    for i, name in enumerate(EMOTIONS):
        print(f"  {name:10s}: {class_weights[i].item():.4f}")

    # 2. Instantiate CNN architecture
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    # 3. Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Adam optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    # 5. Training loop
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation phase (deterministic evaluation)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_acc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))
        history["lr"].append(current_lr)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
            f"LR: {current_lr:.1e}",
            flush=True
        )

        # Save best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> [SAVED BEST MODEL] New highest Val Acc: {best_val_acc:.2f}% at {BEST_MODEL_PATH}", flush=True)

    total_time = time.time() - start_time
    print("=" * 65, flush=True)
    print(f"TRAINING COMPLETE in {total_time / 60:.2f} minutes.", flush=True)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"Final Model Saved: {BEST_MODEL_PATH}")

    # Ensure class_names.json exists
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(EMOTIONS, f, indent=2)
    print(f"Class names verified at: {CLASS_NAMES_PATH}")

    # Save training history
    history_path = BEST_MODEL_PATH.parent / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    print("=" * 65)

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train custom CNN on FER-2013")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
