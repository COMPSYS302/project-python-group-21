import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # Define the first convolutional layer with 1 input channel and 64 output channels, kernel size 3, and padding 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Define the second convolutional layer with 64 input channels and 64 output channels, kernel size 3, and padding 1
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Define the max pooling layer with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the first fully connected layer
        self.fc1 = nn.Linear(64 * 14 * 14, 1000)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(1000, 1000)
        # Define the third fully connected layer with the number of classes as output
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation
        x = F.relu(self.conv1_1(x))
        # Apply the second convolutional layer followed by ReLU activation
        x = F.relu(self.conv1_2(x))
        # Apply max pooling
        x = self.pool(x)

        # Flatten the tensor (keeping the batch size dimension intact)
        x = torch.flatten(x, 1)
        return x  # Return the flattened feature map

    def forward_full(self, x):
        # Get the feature map from the forward method
        x = self.forward(x)
        # Pass through the first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Pass through the second fully connected layer followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Pass through the third fully connected layer to get the final output
        x = self.fc3(x)

        return x  # Return the final output
