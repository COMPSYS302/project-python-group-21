import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)

        # 1x1 convolution followed by two 3x3 convolutions branch
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        # Average pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        # Forward pass through each branch
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Concatenate outputs from all branches
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):
    def __init__(self, num_classes=36):
        super(InceptionV3, self).__init__()
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception module
        self.inception_a = InceptionA(in_channels=64)

        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * 4, num_classes)

    def forward(self, x):
        # Forward pass through initial convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        # Forward pass through Inception module
        x = self.inception_a(x)

        # Forward pass through pooling and fully connected layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def build_inception_v3(num_classes=36):
    return InceptionV3(num_classes=num_classes)
