import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, TEST_PATH, EMOTIONS
from models.emotion_cnn import EmotionCNN


def extract_and_visualize_feature_maps(emotion_sample: str = "happy", output_file: str = "feature_maps_analysis.png"):
    """
    Extracts and visualizes intermediate CNN feature maps from Conv1, Conv2, and Conv3 layers.
    Demonstrates layer-by-layer hierarchical feature abstraction on an actual FER-2013 image.
    """
    print("=" * 65)
    print("INTERMEDIATE CNN FEATURE MAP EXTRACTION & ANALYSIS")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load trained model (or fresh architecture if weights not yet saved)
    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    if BEST_MODEL_PATH.exists():
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded trained weights from {BEST_MODEL_PATH}")
    else:
        print("[WARN] Trained model weights not found, using initialized weights.")

    model.eval()

    # 2. Pick a sample test image from the specified emotion
    sample_dir = TEST_PATH / emotion_sample
    test_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not test_images:
        raise FileNotFoundError(f"No test images found in {sample_dir}")

    image_path = test_images[0]
    print(f"Sample test image: {image_path} (True Emotion: '{emotion_sample}')")

    # 3. Preprocess image
    pil_img = Image.open(image_path).convert("L")
    input_arr = np.array(pil_img, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Extract activations via direct sub-module passes
    with torch.no_grad():
        # Conv 1 output
        conv1_out = model.relu(model.bn1(model.conv1(input_tensor)))
        pool1_out = model.pool1(conv1_out)

        # Conv 2 output
        conv2_out = model.relu(model.bn2(model.conv2(pool1_out)))
        pool2_out = model.pool2(conv2_out)

        # Conv 3 output
        conv3_out = model.relu(model.bn3(model.conv3(pool2_out)))

    # Convert to CPU numpy for visualization
    fmaps_conv1 = conv1_out.squeeze(0).cpu().numpy()  # shape: [64, 48, 48]
    fmaps_conv2 = conv2_out.squeeze(0).cpu().numpy()  # shape: [128, 24, 24]
    fmaps_conv3 = conv3_out.squeeze(0).cpu().numpy()  # shape: [256, 12, 12]

    print(f"Conv 1 Feature Maps Shape: {fmaps_conv1.shape} (Channels: 64, Size: 48x48)")
    print(f"Conv 2 Feature Maps Shape: {fmaps_conv2.shape} (Channels: 128, Size: 24x24)")
    print(f"Conv 3 Feature Maps Shape: {fmaps_conv3.shape} (Channels: 256, Size: 12x12)")

    # 5. Create multi-row visualization grid
    num_cols = 8
    fig, axes = plt.subplots(4, num_cols, figsize=(16, 9))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Row 0: Original input image in first column, rest blank
    axes[0, 0].imshow(input_arr, cmap="gray")
    axes[0, 0].set_title(f"Input: {emotion_sample}", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")
    for j in range(1, num_cols):
        axes[0, j].axis("off")
    fig.text(0.5, 0.92, "Input Face Image (48x48 Grayscale)", ha="center", fontsize=12, fontweight="bold")

    # Row 1: First 8 channels of ConvBlock 1 (Edges & low-level contrast)
    for j in range(num_cols):
        axes[1, j].imshow(fmaps_conv1[j], cmap="viridis")
        axes[1, j].set_title(f"C1 Ch {j+1}", fontsize=9)
        axes[1, j].axis("off")

    # Row 2: First 8 channels of ConvBlock 2 (Mid-level facial parts: mouth, eyes)
    for j in range(num_cols):
        axes[2, j].imshow(fmaps_conv2[j], cmap="plasma")
        axes[2, j].set_title(f"C2 Ch {j+1}", fontsize=9)
        axes[2, j].axis("off")

    # Row 3: First 8 channels of ConvBlock 3 (High-level abstract emotion patterns)
    for j in range(num_cols):
        axes[3, j].imshow(fmaps_conv3[j], cmap="inferno")
        axes[3, j].set_title(f"C3 Ch {j+1}", fontsize=9)
        axes[3, j].axis("off")

    # Section Labels on Left
    fig.text(0.08, 0.70, "Conv 1 (Low-level Edges)", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.46, "Conv 2 (Facial Features)", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    fig.text(0.08, 0.22, "Conv 3 (Emotion Semantics)", va="center", rotation="vertical", fontsize=11, fontweight="bold")

    # Save visualization
    save_path = PROJECT_ROOT / "models" / output_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\n[SUCCESS] Feature map visualization saved to: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    extract_and_visualize_feature_maps()
