import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import VGG16
from resnet import ResNet18


class SignSysModel(nn.Module):
    def __init__(self, vgg, resnet, num_classes=10):
        super(SignSysModel, self).__init__()
        self.vgg = vgg
        self.resnet = resnet

        vgg_output_dim = 64 * 14 * 14
        resnet_output_dim = 512

        self.fc1 = nn.Linear(vgg_output_dim + resnet_output_dim, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        vgg_features = self.vgg(x)
        resnet_features = self.resnet(x)

        # Flatten the VGG features
        vgg_features = torch.flatten(vgg_features, 1)

        # Concatenating the features of the CNNs
        combined_features = torch.cat((vgg_features, resnet_features), dim=1)

        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
