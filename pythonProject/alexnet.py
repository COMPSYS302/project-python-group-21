import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=36):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization for the first conv layer
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer

            # Second convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization for the second conv layer
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer

            # Third convolutional layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Batch normalization for the third conv layer
            nn.ReLU(inplace=True),  # Activation function

            # Fourth convolutional layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Batch normalization for the fourth conv layer
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(256 * 3 * 3, 1024),  # Fully connected layer
            nn.BatchNorm1d(1024),  # Batch normalization for the first FC layer
            nn.ReLU(inplace=True),  # Activation function
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(1024, 512),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization for the second FC layer
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(512, num_classes),  # Output layer
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through feature extraction layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)  # Pass flattened tensor through classifier
        return x


def build_alexnet(num_classes=36):
    # Function to build the AlexNet model
    return AlexNet(num_classes=num_classes)
