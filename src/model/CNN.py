
from src.model.lightning_base_model import LitBaseModel
import torch
import torch.nn as nn

class SimpleCNN(LitBaseModel):
    def __init__(self, params):
        super(SimpleCNN, self).__init__(params)
        self.conv1 = nn.Conv2d(3, 6, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
