import torch
import torch.nn as nn
import torch.nn.functional as F



class SignSysModel(nn.Module):
    def __init__(self, vgg, resnet, num_classes=10):
        super(SignSysModel, self).__init__()
        self.vgg = vgg  # VGG16 model instance
        self.resnet = resnet  # ResNet18 model instance

        # Output dimensions after the VGG and ResNet layers
        vgg_output_dim = 64 * 14 * 14  # VGG output dimension after flattening
        resnet_output_dim = 512  # ResNet output dimension

        # Fully connected layers
        self.fc1 = nn.Linear(vgg_output_dim + resnet_output_dim, 1000)  # First fully connected layer
        self.fc2 = nn.Linear(1000, 1000)  # Second fully connected layer
        self.fc3 = nn.Linear(1000, num_classes)  # Final output layer

    def forward(self, x):
        vgg_features = self.vgg(x)  # Get features from VGG model
        resnet_features = self.resnet(x)  # Get features from ResNet model

        # Flatten the VGG features (reshape to (batch_size, -1))
        vgg_features = torch.flatten(vgg_features, 1)

        # Concatenate the features from both CNNs along the feature dimension
        combined_features = torch.cat((vgg_features, resnet_features), dim=1)

        # Pass the combined features through the fully connected layers with ReLU activations
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final output layer (no activation, usually followed by a loss function like CrossEntropyLoss)

        return x  # Return the output
