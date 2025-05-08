import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # conv1_64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # conv2_64
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # maxpool_1

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # conv3_128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # conv4_128
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # maxpool_2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # conv5_256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv6_256
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # maxpool_3

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # conv7_512
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        return self.conv(x)


class Classifier(nn.Module):
    def __init__(self, in_channels=512*4*4, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_channels, 2048),  # fc1_2048
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),  # fc2_1024
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.fc(x)
