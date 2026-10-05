import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, EMOTIONS
from backend.model_loader import load_model, preprocess_image
from data_pipeline import load_split_paths


def run_feature_map_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading trained model from {BEST_MODEL_PATH} for Feature Map Analysis...")
    model = load_model(str(BEST_MODEL_PATH), device=device)
    model.eval()

    # Select a distinct validation image (e.g. happy or surprise with clear facial landmarks)
    _, (val_p, val_y), _ = load_split_paths()
    sample_idx = None
    for i, y in enumerate(val_y):
        if EMOTIONS[y] == "happy":
            sample_idx = i
            break
    if sample_idx is None:
        sample_idx = 0

    sample_path = val_p[sample_idx]
    actual_emotion = EMOTIONS[val_y[sample_idx]]
    print(f"[INFO] Analyzing validation image: {sample_path.name} (Actual Emotion: {actual_emotion})")

    tensor = preprocess_image(sample_path).to(device)

    # Capture layer activations
    with torch.no_grad():
        activations = model.get_feature_maps(tensor)

    b1_act = activations["block1"].squeeze(0).cpu().numpy() # [64, 48, 48]
    b2_act = activations["block2"].squeeze(0).cpu().numpy() # [128, 24, 24]
    b3_act = activations["block3"].squeeze(0).cpu().numpy() # [256, 12, 12]

    # Create visualization figure
    num_channels_to_show = 8
    fig, axes = plt.subplots(4, num_channels_to_show, figsize=(16, 9))

    # Row 0: Original image in first slot, info in others
    input_img = tensor.squeeze().cpu().numpy()
    for col in range(num_channels_to_show):
        axes[0, col].axis("off")
    axes[0, 0].imshow(input_img, cmap="gray")
    axes[0, 0].set_title(f"Input Face: {actual_emotion.upper()}", fontsize=11, fontweight="bold")

    # Row 1: Block 1 Channels (Low-level edge & contrast detectors)
    for col in range(num_channels_to_show):
        axes[1, col].imshow(b1_act[col], cmap="viridis")
        axes[1, col].set_title(f"Block 1 Ch {col}", fontsize=9)
        axes[1, col].axis("off")

    # Row 2: Block 2 Channels (Mid-level facial parts: eyes, mouth contours)
    for col in range(num_channels_to_show):
        axes[2, col].imshow(b2_act[col], cmap="viridis")
        axes[2, col].set_title(f"Block 2 Ch {col}", fontsize=9)
        axes[2, col].axis("off")

    # Row 3: Block 3 Channels (High-level abstract emotional expression patterns)
    for col in range(num_channels_to_show):
        axes[3, col].imshow(b3_act[col], cmap="magma")
        axes[3, col].set_title(f"Block 3 Ch {col}", fontsize=9)
        axes[3, col].axis("off")

    fig.suptitle(
        f"Layer-by-Layer Feature Map Activation Hierarchy (FER-2013: {actual_emotion.upper()})\n"
        "Block 1: Edges & Textures | Block 2: Eyes & Mouth Geometries | Block 3: Holistic Expression Features",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    output_path = PROJECT_ROOT / "models" / "feature_maps_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Feature map visualization saved to: {output_path}")

    # Theoretical Interpretation for viva / report
    print("\n" + "=" * 65)
    print("FEATURE-MAP HIERARCHICAL INTERPRETATION REPORT")
    print("=" * 65)
    print("1. BLOCK 1 (Layers Conv1_1, Conv1_2 - 64 filters, 48x48):")
    print("   - High spatial resolution preserved.")
    print("   - Activations prominently fire on sharp localized pixel gradients:")
    print("     directional edges, lighting boundaries, and skin texture transitions.")
    print("\n2. BLOCK 2 (Layers Conv2_1, Conv2_2 - 128 filters, 24x24):")
    print("   - Spatial resolution reduced via MaxPool2d (receptive field expanded).")
    print("   - Activations combine edge primitives into localized facial components:")
    print("     nasolabial folds, lip curvature, eye sockets, and eyebrow angles.")
    print("\n3. BLOCK 3 (Layers Conv3_1, Conv3_2 - 256 filters, 12x12):")
    print("   - Most abstract representation capturing semantic facial configurations.")
    print("   - Channels respond specifically to expression archetypes:")
    print("     upturned mouth corners (happiness), open jaw/eyes (surprise),")
    print("     or compressed eyebrow regions (anger/disgust).")
    print("=" * 65)


if __name__ == "__main__":
    run_feature_map_analysis()
