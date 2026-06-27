import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from src.model.lightning_base_model import LitBaseModel

class ResNet50Pretrained(LitBaseModel):
    """
    Ein ResNet-50 mit vortrainiertem Backbone für Bildklassifikation in PyTorch Lightning.

    Erwartet folgende params im YAML:
      model:
        model_params:
          pretrained (bool): Ob vortrainierte Gewichte geladen werden (Default: True)
          num_classes (int): Anzahl der Ausgabeklassen
          freeze_backbone (bool): Backbone einfrieren? (Default: False)
          dropout (float): Dropout-Wahrscheinlichkeit für den Head (Default: 0.0)
          head_hidden_size (int, optional): Grösse einer versteckten Schicht im Head
    """
    def __init__(self, params):
        super().__init__(params)
        cfg = params.get('model', {}).get('model_params', {})
        pretrained = cfg.get('pretrained', True)
        num_classes = cfg['num_classes']
        freeze_backbone = cfg.get('freeze_backbone', False)
        dropout = cfg.get('dropout', 0.0)
        hidden_size = cfg.get('head_hidden_size', None)

        # load pretrained weights, or not
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        # build classification head dynamically
        in_features = self.model.fc.in_features
        layers = []
        if hidden_size:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, num_classes))
        else:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_features, num_classes))
        self.model.fc = nn.Sequential(*layers)

        # initialize classifier
        classifier = self.model.fc[-1]
        if isinstance(classifier, nn.Linear):
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(classifier.bias)

        # freeze backbone (including BatchNorm in eval mode)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        # safe hyperparameters for lightning logging
        self.save_hyperparameters(params)

    def forward(self, x):
        """
        Forward-Pass durch ResNet-50 Backbone und neuen Head.

        Args:
            x (Tensor): Eingabe-Tensor der Form (B, C, H, W)
        Returns:
            Tensor: Logits der Form (B, num_classes)
        """
        return self.model(x)
