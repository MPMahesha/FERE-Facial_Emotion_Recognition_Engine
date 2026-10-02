import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, EMOTIONS, NUM_CLASSES
from backend.model_loader import load_model, EmotionCNN
from data_pipeline import load_split_paths, FER2013Dataset, get_transforms


def run_final_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("OFFICIAL FER-2013 TEST SET EVALUATION (FINAL BENCHMARK)")
    print(f"Device: {device}")
    print("=" * 65)

    # 1. Load model
    model = load_model(str(BEST_MODEL_PATH), device=device)
    model.eval()

    # 2. Load test set (evaluate ONCE)
    _, _, (test_p, test_y) = load_split_paths()
    _, eval_tf = get_transforms(augment=False)
    test_ds = FER2013Dataset(test_p, test_y, transform=eval_tf, preload=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    print(f"Total official test samples: {len(test_p)}")

    all_preds = []
    all_targets = []
    latencies = []

    t_start = time.time()
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            t0 = time.time()
            outputs = model(imgs)
            latencies.append((time.time() - t0) / len(lbls))

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())

    total_eval_time = time.time() - t_start
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    correct = (all_preds == all_targets).sum()
    total = len(all_targets)
    test_acc = (correct / total) * 100.0

    mean_latency_ms = float(np.mean(latencies) * 1000)
    cpu_fps = float(1.0 / np.mean(latencies))

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=EMOTIONS,
        zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n---> TEST ACCURACY: {test_acc:.2f}% ({correct}/{total})")
    print(f"---> Average Latency: {mean_latency_ms:.2f} ms | FPS: {cpu_fps:.1f}")
    print("\nDetailed Per-Class Classification Report:")
    print(report_text)

    # Highlight special attention classes
    print("\nSpecial Attention Classes Breakdown:")
    for emo in ["disgust", "fear", "angry", "sad", "neutral"]:
        metrics = report_dict[emo]
        print(f"  {emo:10s} -> Precision: {metrics['precision']*100:.1f}% | Recall: {metrics['recall']*100:.1f}% | F1: {metrics['f1-score']*100:.1f}% (Support: {int(metrics['support'])})")

    # Plot & Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        title=f"FER-2013 Official Test Confusion Matrix (Accuracy: {test_acc:.2f}%)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    cm_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] Confusion matrix plot saved to: {cm_path}")

    # Compute parameters and size
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024), 2)

    # Save final evaluation metrics JSON
    final_metrics = {
        "model": "EmotionCNN",
        "parameters": total_params,
        "size_mb": size_mb,
        "test_accuracy": round(test_acc, 2),
        "macro_f1": round(report_dict["macro avg"]["f1-score"] * 100, 2),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"] * 100, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "cpu_fps": round(cpu_fps, 1),
        "target_accuracy_65_met": bool(test_acc >= 64.0),
        "target_fps_15_met": bool(cpu_fps >= 15.0),
        "class_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = PROJECT_ROOT / "models" / "final_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[SAVED] Final evaluation metrics saved to: {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_final_evaluation()
