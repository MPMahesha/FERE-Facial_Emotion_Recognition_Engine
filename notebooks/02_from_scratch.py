import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# MANUAL 2D CONVOLUTION WITH NUMPY (FROM SCRATCH)
# ============================================================

def conv2d_from_scratch(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Performs 2D convolution using raw NumPy and explicit nested loops.
    Demonstrates the mathematical mechanics of cross-correlation / convolution:
    output[i, j] = sum(image_patch * kernel)
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # Apply zero-padding if specified
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant", constant_values=0)
    else:
        padded_image = image

    pad_H, pad_W = padded_image.shape

    out_H = (pad_H - kH) // stride + 1
    out_W = (pad_W - kW) // stride + 1

    output = np.zeros((out_H, out_W), dtype=np.float32)

    # Slide kernel over image
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW

            patch = padded_image[h_start:h_end, w_start:w_end]
            # Element-wise product followed by summation
            output[i, j] = np.sum(patch * kernel)

    return output


def demo():
    print("=" * 60)
    print("DEMO: MANUAL 2D CONVOLUTION (FROM SCRATCH vs PYTORCH)")
    print("=" * 60)

    # 1. Create a sample 6x6 grayscale matrix
    image = np.array([
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0]
    ], dtype=np.float32)

    # 2. Create a standard 3x3 vertical edge Sobel-style filter
    kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    print("Input Image (6x6):")
    print(image.astype(int))
    print("\nKernel (3x3 Vertical Edge Detector):")
    print(kernel.astype(int))

    # 3. Manual NumPy convolution (stride=1, padding=1)
    numpy_output = conv2d_from_scratch(image, kernel, stride=1, padding=1)
    print("\nNumPy Output (with padding=1):")
    print(numpy_output)

    # 4. Compare with PyTorch Conv2D
    # PyTorch expects shape: [Batch, Channel, Height, Width]
    tensor_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor_kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)

    # F.conv2d in PyTorch implements cross-correlation (same as standard CNNs)
    torch_output = F.conv2d(tensor_img, tensor_kernel, stride=1, padding=1).squeeze().numpy()

    print("\nPyTorch F.conv2d Output:")
    print(torch_output)

    diff = np.max(np.abs(numpy_output - torch_output))
    print(f"\nMaximum difference between NumPy and PyTorch: {diff:.6f}")
    assert diff < 1e-5, "Mismatch between NumPy and PyTorch convolution!"
    print("[SUCCESS] Manual 2D convolution matches PyTorch exactly.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
