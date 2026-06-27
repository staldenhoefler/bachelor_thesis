from src.model.lightning_base_model import LitBaseModel
import torch
import torch.nn as nn

class BeforeAfterManyFeatureMapsCNN(LitBaseModel):
    def __init__(self, params):
        super(BeforeAfterManyFeatureMapsCNN, self).__init__(params)
        self.conv1 = nn.Conv2d(6, 32, 7)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 7)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=params['model']['model_params']['dropout'])
        self.fc1 = nn.LazyLinear(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 120)
        self.bn6 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, 6)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.bn4(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn5(torch.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn6(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
