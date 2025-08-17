import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

class FeatureGenerator(nn.Module):
    def __init__(self, num_channals, image_size=32):
        super(FeatureGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channals[0], kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(num_channals[0], num_channals[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_channals[0])
        self.conv2 = nn.Conv2d(num_channals[0], num_channals[1], kernel_size=5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(num_channals[1], num_channals[1], kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_channals[1])
        self.conv3 = nn.Conv2d(num_channals[1], num_channals[2], kernel_size=5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(num_channals[2], num_channals[2], kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(num_channals[2])
        self.flat_dim = int(num_channals[2]*(image_size*image_size/16))
        self.fc1 = nn.Linear(self.flat_dim, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.bn1(self.conv1_2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2_2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), self.flat_dim)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes=10, prob=0.5):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def ModelFactory(device, num_classes=10, image_size=32):
    model_G = FeatureGenerator(num_channals=[64, 64, 128], image_size=image_size).to(device)
    model_F1 = Classifier(num_classes=num_classes).to(device)
    model_F2 = Classifier(num_classes=num_classes).to(device)
    return model_G,model_F1,model_F2