import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CUSTOM VGG-STYLE CNN FOR 48x48 GRAYSCALE EMOTION RECOGNITION
# ============================================================

class EmotionCNN(nn.Module):
    """
    Custom VGG-style CNN designed for 48x48 Grayscale Facial Emotion Recognition.
    Architecture:
        - ConvBlock 1: Conv2d(1 -> 64)  -> BatchNorm -> ReLU -> MaxPool (48x48 -> 24x24)
        - ConvBlock 2: Conv2d(64 -> 128) -> BatchNorm -> ReLU -> MaxPool (24x24 -> 12x12)
        - ConvBlock 3: Conv2d(128 -> 256)-> BatchNorm -> ReLU -> MaxPool (12x12 -> 6x6)
        - Classifier:  Flatten -> Linear(256*6*6 -> 512) -> ReLU -> Dropout(0.5) -> Linear(512 -> 7)
    """
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()

        # ConvBlock 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvBlock 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ConvBlock 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense Classifier
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvBlock 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # ConvBlock 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # ConvBlock 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Dense Classification
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def get_model_summary(model: nn.Module) -> dict:
    """
    Computes parameter counts and estimated model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "size_mb": round(size_mb, 2)
    }


if __name__ == "__main__":
    net = EmotionCNN(num_classes=7)
    dummy_input = torch.randn(1, 1, 48, 48)
    output = net(dummy_input)

    summary = get_model_summary(net)
    print("EmotionCNN Architecture:")
    print(net)
    print(f"\nDummy forward pass output shape: {output.shape} (Expected: [1, 7])")
    print(f"Total Parameters:    {summary['total_parameters']:,}")
    print(f"Trainable Parameters:{summary['trainable_parameters']:,}")
    print(f"Model Size (weights): {summary['size_mb']} MB")
