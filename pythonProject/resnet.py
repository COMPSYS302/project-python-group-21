import torchvision.models as models
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Load a pre-trained ResNet-18 model from torchvision
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept single-channel (grayscale) images instead of three-channel (RGB) images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the fully connected layer with an identity function to remove it (useful for feature extraction)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        # Pass the input through the ResNet model
        x = self.resnet(x)
        return x  # Return the output (features from the ResNet model)
