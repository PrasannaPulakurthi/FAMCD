import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # conv1_64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # conv2_128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # conv3_256
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.conv(x)


class Classifier(nn.Module):
    def __init__(self, in_channels=256*4*4, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 2048),  # fc1_2048
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),  # fc2_512
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.fc(x)
