import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import torch
from dataloader import UDADataLoader
from tqdm import tqdm
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32, 32, 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32, 16, 16
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64, 16, 16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64, 16, 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64, 8, 8
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.fc(x)
    
class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)
