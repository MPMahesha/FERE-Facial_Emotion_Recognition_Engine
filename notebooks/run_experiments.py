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
from torch.utils.data import DataLoader, Subset

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
from models.emotion_cnn import EmotionCNN, get_model_summary
from data_pipeline import get_dataloaders, load_split_paths, FER2013Dataset, get_transforms, compute_class_weights


# Baseline 3-conv CNN (as previously in baseline Experiment A)
class BaselineEmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 5,
    device: torch.device = None
):
    print(f"\n---> Starting {exp_name} ({epochs} epochs)...")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lbls)
            preds = out.argmax(dim=1)
            train_correct += (preds == lbls).sum().item()
            train_total += len(lbls)

        ep_train_loss = train_loss / train_total
        ep_train_acc = (train_correct / train_total) * 100.0

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item() * len(lbls)
                preds = out.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total += len(lbls)

        ep_val_loss = val_loss / val_total
        ep_val_acc = (val_correct / val_total) * 100.0

        history["train_loss"].append(round(ep_train_loss, 4))
        history["train_acc"].append(round(ep_train_acc, 2))
        history["val_loss"].append(round(ep_val_loss, 4))
        history["val_acc"].append(round(ep_val_acc, 2))

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc

        print(f"  [{exp_name}] Epoch {epoch:02d}/{epochs:02d} | Train Acc: {ep_train_acc:.2f}% | Val Acc: {ep_val_acc:.2f}% | Val Loss: {ep_val_loss:.4f}")

    return {
        "final_train_acc": history["train_acc"][-1],
        "best_val_acc": best_val_acc,
        "final_val_loss": history["val_loss"][-1],
        "history": history
    }


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running controlled experiments suite on {device}")

    # Load paths
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = load_split_paths(validation_split=0.15, random_seed=RANDOM_SEED)

    # Use a stratified 5,000-image benchmark subset for fast, fair ablation comparison
    rng = random.Random(RANDOM_SEED)
    sample_indices = []
    # Stratified 700 per class (~4900 total)
    for class_idx in range(NUM_CLASSES):
        cls_idx = [i for i, y in enumerate(train_y) if y == class_idx]
        sample_indices.extend(rng.sample(cls_idx, min(700, len(cls_idx))))
    rng.shuffle(sample_indices)

    sub_train_p = [train_p[i] for i in sample_indices]
    sub_train_y = [train_y[i] for i in sample_indices]

    _, eval_tf = get_transforms(augment=False)
    val_dataset = FER2013Dataset(val_p, val_y, transform=eval_tf)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    experiments = []

    # -------------------------------------------------------------
    # Experiment A: Baseline Record (from history)
    # -------------------------------------------------------------
    exp_a_summary = {
        "experiment": "Experiment A (Baseline)",
        "description": "Old 3-conv CNN + Raw Inverse Weights (Disgust=9.42) + Basic Augmentation",
        "parameters": 5093255,
        "size_mb": 19.43,
        "train_acc": 30.14,
        "val_acc": 39.68,
        "val_loss": 1.6608,
        "configuration": "Adam lr=1e-3, Raw inverse class weights, 15 epochs"
    }
    experiments.append(exp_a_summary)
    print("\n[RECORDED] Experiment A (Baseline): Val Acc = 39.68%, Train Acc = 30.14%")

    # -------------------------------------------------------------
    # Experiment B: Corrected Pipeline (Unweighted Loss, Clean Preprocessing, No Aug)
    # -------------------------------------------------------------
    set_seed(RANDOM_SEED)
    train_tf_b, _ = get_transforms(augment=False)
    ds_b = FER2013Dataset(sub_train_p, sub_train_y, transform=train_tf_b)
    loader_b = DataLoader(ds_b, batch_size=128, shuffle=True)

    model_b = BaselineEmotionCNN(num_classes=7).to(device)
    crit_b = nn.CrossEntropyLoss() # Unweighted
    opt_b = optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=1e-4)

    res_b = run_single_experiment("Exp B (Corrected Pipeline, Unweighted)", model_b, loader_b, val_loader, crit_b, opt_b, epochs=5, device=device)
    experiments.append({
        "experiment": "Experiment B (Corrected Pipeline)",
        "description": "Baseline CNN + Clean Preprocessing + Unweighted Loss (No extreme disgust weight)",
        "parameters": 5093255,
        "size_mb": 19.43,
        "train_acc": res_b["final_train_acc"],
        "val_acc": res_b["best_val_acc"],
        "val_loss": res_b["final_val_loss"],
        "configuration": "Adam lr=1e-3, Unweighted CrossEntropyLoss, No Aug"
    })

    # -------------------------------------------------------------
    # Experiment C: Corrected Pipeline + Sqrt Damped Class Weights
    # -------------------------------------------------------------
    set_seed(RANDOM_SEED)
    weights_c = compute_class_weights(sub_train_y, num_classes=7, method="sqrt").to(device)
    model_c = BaselineEmotionCNN(num_classes=7).to(device)
    crit_c = nn.CrossEntropyLoss(weight=weights_c)
    opt_c = optim.Adam(model_c.parameters(), lr=1e-3, weight_decay=1e-4)

    res_c = run_single_experiment("Exp C (Pipeline + Sqrt Class Weights)", model_c, loader_b, val_loader, crit_c, opt_c, epochs=5, device=device)
    experiments.append({
        "experiment": "Experiment C (Pipeline + Sqrt Weights)",
        "description": "Baseline CNN + Sqrt Damped Class Weights (Disgust ~2.4x)",
        "parameters": 5093255,
        "size_mb": 19.43,
        "train_acc": res_c["final_train_acc"],
        "val_acc": res_c["best_val_acc"],
        "val_loss": res_c["final_val_loss"],
        "configuration": "Adam lr=1e-3, Sqrt Damped Class Weights, No Aug"
    })

    # -------------------------------------------------------------
    # Experiment D: Corrected Pipeline + Realistic Facial Augmentation
    # -------------------------------------------------------------
    set_seed(RANDOM_SEED)
    train_tf_d, _ = get_transforms(augment=True)
    ds_d = FER2013Dataset(sub_train_p, sub_train_y, transform=train_tf_d)
    loader_d = DataLoader(ds_d, batch_size=128, shuffle=True)

    model_d = BaselineEmotionCNN(num_classes=7).to(device)
    crit_d = nn.CrossEntropyLoss(weight=weights_c)
    opt_d = optim.Adam(model_d.parameters(), lr=1e-3, weight_decay=1e-4)

    res_d = run_single_experiment("Exp D (Pipeline + Weights + Aug)", model_d, loader_d, val_loader, crit_d, opt_d, epochs=5, device=device)
    experiments.append({
        "experiment": "Experiment D (Pipeline + Weights + Aug)",
        "description": "Baseline CNN + Sqrt Weights + Realistic Facial Augmentation",
        "parameters": 5093255,
        "size_mb": 19.43,
        "train_acc": res_d["final_train_acc"],
        "val_acc": res_d["best_val_acc"],
        "val_loss": res_d["final_val_loss"],
        "configuration": "Adam lr=1e-3, Sqrt Class Weights, Realistic Augmentation"
    })

    # -------------------------------------------------------------
    # Experiment E: Final Improved VGG-Style CNN (Deep 6-Conv + Adaptive Pool)
    # -------------------------------------------------------------
    set_seed(RANDOM_SEED)
    model_e = EmotionCNN(num_classes=7).to(device)
    opt_e = optim.AdamW(model_e.parameters(), lr=1e-3, weight_decay=1e-3)
    crit_e = nn.CrossEntropyLoss(weight=weights_c)

    res_e = run_single_experiment("Exp E (Improved VGG CNN Configuration)", model_e, loader_d, val_loader, crit_e, opt_e, epochs=5, device=device)
    summary_e = get_model_summary(model_e)
    experiments.append({
        "experiment": "Experiment E (Final Improved CNN)",
        "description": "6-Conv VGG CNN + BatchNorm + Spatial Pool + Sqrt Weights + Augmentation",
        "parameters": summary_e["total_parameters"],
        "size_mb": summary_e["size_mb"],
        "train_acc": res_e["final_train_acc"],
        "val_acc": res_e["best_val_acc"],
        "val_loss": res_e["final_val_loss"],
        "configuration": "AdamW lr=1e-3, Sqrt Weights, Augmentation, 6-Conv VGG Architecture"
    })

    # Save experiments comparison
    out_path = BEST_MODEL_PATH.parent / "experiments_comparison.json"
    with open(out_path, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"\n[SUCCESS] Experiments comparison saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENTS COMPARISON TABLE")
    print("=" * 80)
    for exp in experiments:
        print(f"- {exp['experiment']:<36s} | Val Acc: {exp['val_acc']:5.2f}% | Train Acc: {exp['train_acc']:5.2f}% | Params: {exp['parameters']:,} ({exp['size_mb']} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
