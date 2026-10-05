import os
import sys
import time
import json
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, EMOTIONS, NUM_CLASSES, RANDOM_SEED, BATCH_SIZE
from models.emotion_cnn import EmotionCNN, get_model_summary
from data_pipeline import (
    FER2013Dataset,
    compute_class_weights,
    get_transforms,
    load_split_paths,
)


# ============================================================
# TRANSFER-LEARNING INPUT PREPARATION
# ============================================================

TRANSFER_IMAGE_SIZE = 224


def prepare_transfer_inputs(images: torch.Tensor) -> torch.Tensor:
    """Upscales grayscale FER crops and repeats them to the three RGB channels."""
    images = images.repeat(1, 3, 1, 1)
    return F.interpolate(
        images,
        size=(TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )


# ============================================================
# 1. RESNET-18 (PRETRAINED RGB MODEL)
# ============================================================


def build_resnet18(num_classes: int = 7) -> nn.Module:
    """
    Builds pretrained ResNet-18 with an emotion-classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 2. EFFICIENTNET-B0 (PRETRAINED RGB MODEL)
# ============================================================

def build_efficientnet_b0(num_classes: int = 7) -> nn.Module:
    """
    Builds pretrained EfficientNet-B0 with an emotion-classification head.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 3. SMALL VISION TRANSFORMER (FOR 48x48 GRAYSCALE)
# ============================================================

class SmallViT(nn.Module):
    """
    Compact Vision Transformer tailored for 48x48 Grayscale Facial Emotion Recognition.
    - Patch size: 6x6 -> (48/6)*(48/6) = 64 tokens
    - Embedding dim: 128
    - Depth: 4 Transformer Encoder Layers
    - Heads: 4
    """
    def __init__(self, img_size: int = 48, patch_size: int = 6, in_channels: int = 1, num_classes: int = 7, embed_dim: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # Linear patch projection
        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Patch extraction: [B, C, H, W] -> [B, num_patches, patch_dim]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p * p).permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.transformer(x)
        cls_rep = self.norm(x[:, 0])
        logits = self.head(cls_rep)
        return logits


# ============================================================
# BENCHMARK & COMPARISON FUNCTION
# ============================================================

def run_model_comparison(train_epochs: int = 10):
    """
    Compares Custom CNN vs ResNet-18 vs EfficientNet-B0 vs Small ViT:
    Parameters, size in MB, inference latency, and validation performance.
    """
    print("=" * 70)
    print("TRANSFER LEARNING & ARCHITECTURE COMPARISON EXPERIMENT")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_paths, train_labels), (val_paths, val_labels), _ = load_split_paths()
    train_transform, eval_transform = get_transforms(augment=True)
    train_dataset = FER2013Dataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FER2013Dataset(val_paths, val_labels, transform=eval_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    models_to_test = {
        "Custom EmotionCNN": EmotionCNN(num_classes=NUM_CLASSES),
        "ResNet-18 (Transfer)": build_resnet18(num_classes=NUM_CLASSES),
        "EfficientNet-B0 (Transfer)": build_efficientnet_b0(num_classes=NUM_CLASSES),
        "Small ViT (Vision Transformer)": SmallViT(num_classes=NUM_CLASSES)
    }

    results = []

    for name, model in models_to_test.items():
        model = model.to(device)
        if name == "Custom EmotionCNN":
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        summary = get_model_summary(model)
        is_transfer_model = name in {
            "ResNet-18 (Transfer)",
            "EfficientNet-B0 (Transfer)"
        }
        input_shape = (3, TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE) if is_transfer_model else (1, 48, 48)

        print(f"\n--- Testing: {name} ---")
        print(f"Parameters: {summary['total_parameters']:,} | Size: {summary['size_mb']} MB")

        # Benchmark every architecture on CPU using its actual deployment input shape.
        cpu_model = copy.deepcopy(model).to("cpu").eval()
        dummy_sample = torch.randn(1, *input_shape)
        with torch.no_grad():
            for _ in range(5):
                _ = cpu_model(dummy_sample)

        timings = []
        num_runs = 50
        with torch.inference_mode():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = cpu_model(dummy_sample)
                timings.append((time.perf_counter() - start) * 1000)
        latency_ms = sum(timings) / len(timings)
        cpu_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        del cpu_model

        print(f"Latency: {latency_ms:.2f} ms/frame | Throughput: {cpu_fps:.1f} FPS")

        # Train transfer models on the complete training split; use the saved
        # checkpoint for the custom CNN baseline. The test split is never used here.
        if train_epochs > 0 and name != "Custom EmotionCNN":
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(train_epochs):
                model.train()
                for imgs, lbls in train_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    if is_transfer_model:
                        imgs = prepare_transfer_inputs(imgs)
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, lbls)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch + 1}/{train_epochs} complete for {name}", flush=True)

        # Evaluate the complete validation split.
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                if is_transfer_model:
                    imgs = prepare_transfer_inputs(imgs)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                val_correct += (pred == lbls).sum().item()
                val_total += lbls.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(f"Full validation accuracy: {val_acc:.2f}%")

        results.append({
            "model": name,
            "parameters": summary["total_parameters"],
            "size_mb": summary["size_mb"],
            "latency_ms": round(latency_ms, 2),
            "cpu_fps": round(cpu_fps, 1),
            "validation_accuracy": round(val_acc, 2),
            "training_epochs": train_epochs if name != "Custom EmotionCNN" else 0,
            "input_shape": list(input_shape),
            "pretrained_weights_requested": is_transfer_model
        })

    # Save comparison summary
    save_path = PROJECT_ROOT / "models" / "model_comparison.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ARCHITECTURAL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Params':<12} {'Size(MB)':<10} {'Latency(ms)':<12} {'FPS':<8} {'Val Acc%':<10}")
    print("-" * 82)
    for r in results:
        print(f"{r['model']:<30} {r['parameters']:<12,d} {r['size_mb']:<10.2f} {r['latency_ms']:<12.2f} {r['cpu_fps']:<8.1f} {r['validation_accuracy']:<10.2f}")
    print("=" * 70)
    print(f"[SUCCESS] Comparison logged to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and benchmark FER-2013 architectures")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_model_comparison(train_epochs=args.epochs)
