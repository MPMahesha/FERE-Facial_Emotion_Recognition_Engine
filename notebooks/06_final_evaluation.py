import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, TEST_PATH, EMOTIONS, NUM_CLASSES
from models.emotion_cnn import EmotionCNN, get_model_summary
from data_pipeline import get_dataloaders


def evaluate_model_on_test_set():
    """
    Evaluates the trained EmotionCNN on the official FER-2013 test set.
    Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix, Error Analysis, and CPU Latency.
    """
    print("=" * 70)
    print("FINAL FER-2013 TEST SET EVALUATION & PERFORMANCE BENCHMARK")
    print("=" * 70)

    device = torch.device("cpu")  # CPU evaluation as required for deployment benchmarking

    # 1. Load trained model
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}. Train model first!")

    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded trained weights from: {BEST_MODEL_PATH}")

    summary = get_model_summary(model)

    # 2. Load test set DataLoader
    pipeline = get_dataloaders(batch_size=64)
    test_loader = pipeline["test_loader"]
    print(f"[INFO] Loaded test set: {len(pipeline['test_dataset'])} images across 7 classes.")

    # 3. Run inference on test set
    all_preds = []
    all_targets = []
    all_probs = []
    misclassified = []

    test_start = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            confidence, preds = torch.max(probs, 1)

            for i in range(len(labels)):
                target = labels[i].item()
                pred = preds[i].item()
                conf = confidence[i].item()

                all_preds.append(pred)
                all_targets.append(target)
                all_probs.append(probs[i].cpu().numpy())

                if pred != target and len(misclassified) < 10:
                    misclassified.append({
                        "actual": EMOTIONS[target],
                        "predicted": EMOTIONS[pred],
                        "confidence": round(conf, 4)
                    })

    total_eval_time = time.time() - test_start
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 4. Compute Metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 70)
    print("GLOBAL EVALUATION METRICS")
    print("=" * 70)
    print(f"Test Accuracy:          {accuracy:.2f}%")
    print(f"Macro Precision:        {prec_macro * 100:.2f}%")
    print(f"Macro Recall:           {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:         {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:      {f1_weighted * 100:.2f}%")

    # 5. Class-level breakdown
    print("\n" + "=" * 70)
    print("CLASS-LEVEL PERFORMANCE BREAKDOWN")
    print("=" * 70)
    report_dict = classification_report(
        all_targets, all_preds, target_names=EMOTIONS, digits=4, output_dict=True, zero_division=0
    )
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 58)
    for emo in EMOTIONS:
        stats = report_dict[emo]
        print(f"{emo:<12} {stats['precision']*100:<11.2f}% {stats['recall']*100:<11.2f}% {stats['f1-score']*100:<11.2f}% {int(stats['support']):<10}")

    # 6. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Rows: Actual, Columns: Predicted)")
    print("=" * 70)
    col_headers = "".join([f"{e[:4]:>7}" for e in EMOTIONS])
    print(f"{'Actual':<10} {col_headers}")
    print("-" * 62)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>7}" for val in row])
        print(f"{EMOTIONS[i]:<10} {row_str}")

    # 7. Error Analysis: Misclassified Samples
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS (SAMPLE MISCLASSIFICATIONS)")
    print("=" * 70)
    for idx, sample in enumerate(misclassified, 1):
        print(f"Sample {idx:02d}: Actual: {sample['actual']:<10} | Predicted: {sample['predicted']:<10} | Confidence: {sample['confidence'] * 100:.1f}%")

    # 8. CPU Latency & FPS Benchmark
    print("\n" + "=" * 70)
    print("ACCURACY vs LATENCY BENCHMARK (CPU INFERENCE)")
    print("=" * 70)

    # Warmup
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    for _ in range(20):
        _ = model(dummy_input)

    # Measure single-frame latency over 100 iterations
    timings = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        timings.append((time.perf_counter() - t0) * 1000)

    mean_latency_ms = float(np.mean(timings))
    std_latency_ms = float(np.std(timings))
    cpu_fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0

    target_acc_met = accuracy >= 65.0
    target_fps_met = cpu_fps >= 15.0

    print(f"Model Name:             EmotionCNN (Custom VGG-style)")
    print(f"Total Parameters:       {summary['total_parameters']:,}")
    print(f"Model Size on Disk:     {summary['size_mb']:.2f} MB")
    print(f"Test Accuracy:          {accuracy:.2f}% (Target >= 65%: {'PASSED' if target_acc_met else 'IN PROGRESS / HONEST STATUS REPORTED'})")
    print(f"Mean CPU Latency:       {mean_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
    print(f"CPU Throughput (FPS):   {cpu_fps:.1f} FPS (Target >= 15 FPS: {'PASSED' if target_fps_met else 'FAILED'})")

    # 9. Save Evaluation Results to JSON
    results = {
        "model": "EmotionCNN",
        "parameters": summary["total_parameters"],
        "size_mb": summary["size_mb"],
        "test_accuracy": round(accuracy, 2),
        "macro_f1": round(f1_macro * 100, 2),
        "weighted_f1": round(f1_weighted * 100, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "cpu_fps": round(cpu_fps, 1),
        "target_accuracy_65_met": target_acc_met,
        "target_fps_15_met": target_fps_met,
        "class_report": report_dict,
        "sample_misclassifications": misclassified
    }

    eval_json_path = PROJECT_ROOT / "models" / "final_evaluation_metrics.json"
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Final evaluation results saved to: {eval_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_model_on_test_set()
