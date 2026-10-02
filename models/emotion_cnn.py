import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# VGG-STYLE DEEP CNN FOR 48x48 GRAYSCALE EMOTION RECOGNITION
# ============================================================

class EmotionCNN(nn.Module):
    """
    Standard VGG-style Deep Convolutional Neural Network designed from scratch
    for 48x48 Grayscale Facial Emotion Recognition (FER-2013).

    Architecture:
        - Block 1 (48x48 -> 24x24):
            Conv2d(1 -> 32, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(32 -> 64, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Block 2 (24x24 -> 12x12):
            Conv2d(64 -> 128, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(128 -> 128, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Block 3 (12x12 -> 6x6):
            Conv2d(128 -> 256, 3x3, pad=1) -> BatchNorm2d -> ReLU
            Conv2d(256 -> 256, 3x3, pad=1) -> BatchNorm2d -> ReLU
            MaxPool2d(2x2) -> Dropout(0.25)
        - Classifier:
            AdaptiveAvgPool2d((3, 3)) -> Flatten (2304)
            Linear(256 * 3 * 3 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.5)
            Linear(512 -> num_classes)
    """
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()

        # ConvBlock 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        # ConvBlock 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # ConvBlock 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        # Dense Classifier
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvBlock 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))

        # ConvBlock 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))

        # ConvBlock 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))

        # Dense Classification
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts intermediate feature activations from each convolutional block
        for layer-by-layer interpretability analysis.
        """
        # Block 1 activation (low-level edges and textures)
        b1 = self.relu(self.bn1_1(self.conv1_1(x)))
        b1 = self.relu(self.bn1_2(self.conv1_2(b1)))

        # Block 2 activation (mid-level facial parts: eyes, mouth contours)
        b2 = self.drop1(self.pool1(b1))
        b2 = self.relu(self.bn2_1(self.conv2_1(b2)))
        b2 = self.relu(self.bn2_2(self.conv2_2(b2)))

        # Block 3 activation (high-level facial emotion expressions)
        b3 = self.drop2(self.pool2(b2))
        b3 = self.relu(self.bn3_1(self.conv3_1(b3)))
        b3 = self.relu(self.bn3_2(self.conv3_2(b3)))

        return {
            "block1": b1,
            "block2": b2,
            "block3": b3
        }


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
    print("EmotionCNN Architecture (VGG-style Deep CNN):")
    print(net)
    print(f"\nDummy forward pass output shape: {output.shape} (Expected: [1, 7])")
    print(f"Total Parameters:    {summary['total_parameters']:,}")
    print(f"Trainable Parameters:{summary['trainable_parameters']:,}")
    print(f"Model Size (weights): {summary['size_mb']} MB")
