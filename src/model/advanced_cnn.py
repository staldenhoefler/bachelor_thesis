from src.model.lightning_base_model import LitBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN with 3 convolutional Layer and 4 Fully-Connected Layer
# Batchnorm after each Convolutional Layer
# Dropout after each Fully-Connected Layer with 0.5
# Kaiming-Init for all Convolutional Layer and Linear Layer
# LazyLinear for the first Fully-Connected Layer
class AdvancedCNN(LitBaseModel):
    def __init__(self, params):
        super(AdvancedCNN, self).__init__(params)
        # Convolutional Backbone
        # Input: RGB images (3 channels)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, bias=False)
        # BatchNorm normalizes the 6 channels after each mini-batch
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, bias=False)
        self.bn3   = nn.BatchNorm2d(16)

        # MaxPooling: 2x2 window, stride=2
        self.pool    = nn.MaxPool2d(2, 2)

        # Dropout 50%, to reduce overfitting
        self.dropout = nn.Dropout(p=0.5)

        # 2) Fully-Connected Classifier
        # LazyLinear projects to 240 neurons
        self.fc1 = nn.LazyLinear(240, bias=False)
        self.bn4 = nn.BatchNorm1d(240)
        self.fc2 = nn.Linear(240, 120, bias=False)
        self.bn5 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 60, bias=False)
        self.bn6 = nn.BatchNorm1d(60)
        self.fc4 = nn.Linear(60, 5)

        # Weight initialization for all fixed Convs and Linears
        self._init_weights()

    def _init_weights(self):
        # Kaiming-Init for all Conv2d and Linear (without LazyLinear)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)                 # [B, 3, H, W] → [B, 6, H-4, W-4]
        x = self.bn1(x)                   # Normalize
        x = F.relu(x)                     # nonlinearity
        x = self.pool(x)                  # [B, 6, (H-4)/2, (W-4)/2]

        # Layer 2
        x = self.conv2(x)                 # [B, 6, …] → [B, 16, …]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)                  # spatial size halved again

        # Layer 3
        x = self.conv3(x)                 # [B, 16, …] → [B, 16, …]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)                  

        # Flatten → Fully-Connected
        x = torch.flatten(x, 1)           # [B, 16 * h' * w'] → [B, N_features]

        # FC1: Linear → BatchNorm → ReLU → Dropout
        x = self.fc1(x)                   # N_features → 240
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC2: 240 → 120
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC3: 120 → 60
        x = self.fc3(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC4: 60 → 5 Logits
        x = self.fc4(x)
        return x

    def configure_optimizers(self):
        """
        Akzeptiert sowohl model_cfg['optimizer_params']
        als auch model_cfg['params'] als Fallback für die Lernrate o. Ä.
        """
        model_cfg    = self.params.get('model', {})
        optim_name   = model_cfg.get('optimizer', 'Adam')
        optim_params = model_cfg.get(
            'optimizer_params',
            model_cfg.get('params', {})  
        )

        if optim_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), **optim_params)
        elif optim_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)
        else:
            optimizer = torch.optim.Adam(self.parameters(), **optim_params)

        return optimizer