import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, EMOTIONS, NUM_CLASSES, RANDOM_SEED, BATCH_SIZE
from models.emotion_cnn import EmotionCNN, get_model_summary
from data_pipeline import get_dataloaders


# ============================================================
# 1. RESNET-18 (ADAPTED FOR 1-CHANNEL 48x48 INPUT)
# ============================================================

def build_resnet18(num_classes: int = 7) -> nn.Module:
    """
    Builds ResNet-18 adapted for 48x48 single-channel grayscale input.
    Replaces first conv with 1-channel kernel and final fc with 7 classes.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)

    # Adapt first layer for 1-channel grayscale
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    # Adapt classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 2. EFFICIENTNET-B0 (ADAPTED FOR 1-CHANNEL INPUT)
# ============================================================

def build_efficientnet_b0(num_classes: int = 7) -> nn.Module:
    """
    Builds EfficientNet-B0 adapted for single-channel input and 7 emotion classes.
    """
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        model = models.efficientnet_b0(weights=None)

    # Adapt first conv layer for 1-channel input
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False
    )
    # Adapt classifier
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

def run_model_comparison(train_epochs: int = 1):
    """
    Compares Custom CNN vs ResNet-18 vs EfficientNet-B0 vs Small ViT:
    Parameters, size in MB, inference latency, and validation performance.
    """
    print("=" * 70)
    print("TRANSFER LEARNING & ARCHITECTURE COMPARISON EXPERIMENT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_dataloaders(batch_size=64)
    train_loader = pipeline["train_loader"]
    val_loader = pipeline["val_loader"]
    class_weights = pipeline["class_weights"].to(device)
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
        summary = get_model_summary(model)

        print(f"\n--- Testing: {name} ---")
        print(f"Parameters: {summary['total_parameters']:,} | Size: {summary['size_mb']} MB")

        # Measure CPU inference latency on single sample
        dummy_sample = torch.randn(1, 1, 48, 48).to(device)
        model.eval()
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_sample)

        # Timing
        t0 = time.time()
        num_runs = 50
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_sample)
        latency_ms = ((time.time() - t0) / num_runs) * 1000
        cpu_fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        print(f"Latency: {latency_ms:.2f} ms/frame | Throughput: {cpu_fps:.1f} FPS")

        # Quick validation evaluation (or 1 epoch training if specified)
        if train_epochs > 0:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for step, (imgs, lbls) in enumerate(train_loader):
                if step >= 40:  # Train on a fast subset for comparative dynamic check
                    break
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

        # Validation check
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for step, (imgs, lbls) in enumerate(val_loader):
                if step >= 30:  # Fast representative val check
                    break
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                val_correct += (pred == lbls).sum().item()
                val_total += lbls.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(f"Val Accuracy (Comparative sample): {val_acc:.2f}%")

        results.append({
            "model": name,
            "parameters": summary["total_parameters"],
            "size_mb": summary["size_mb"],
            "latency_ms": round(latency_ms, 2),
            "cpu_fps": round(cpu_fps, 1),
            "val_acc_sample": round(val_acc, 2)
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
        print(f"{r['model']:<30} {r['parameters']:<12,d} {r['size_mb']:<10.2f} {r['latency_ms']:<12.2f} {r['cpu_fps']:<8.1f} {r['val_acc_sample']:<10.2f}")
    print("=" * 70)
    print(f"[SUCCESS] Comparison logged to {save_path}")


if __name__ == "__main__":
    run_model_comparison(train_epochs=1)
