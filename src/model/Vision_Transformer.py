from torch import nn
from transformers import ViTForImageClassification
from src.model.lightning_base_model import LitBaseModel

class VisionTransformerPretrained(LitBaseModel):
    """
    Vision Transformer (ViT) model with a pretrained backbone for rectangular images,
    implemented in PyTorch Lightning.

    This class uses a pretrained ViT model from Hugging Face's Transformers library and
    replaces its classifier head with a custom architecture. The backbone can optionally
    be frozen to prevent its weights from being updated during training.

    Attributes
    ----------
    num_classes : int
        Number of output classes for classification.
    freeze_backbone : bool
        If True, the backbone parameters are frozen and not updated during training.
    backbone : transformers.ViTForImageClassification
        The pretrained Vision Transformer model used as a backbone.
    in_features : int
        Number of input features for the first layer of the classifier.
    dropout_rate : float
        Dropout rate applied between fully connected layers in the classifier.
    """
    def __init__(self, params):
        super().__init__(params)
        cfg = params['model']['model_params']
        self.num_classes = cfg['num_classes']
        self.freeze_backbone = cfg.get('freeze_backbone', False)

        self.backbone = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-384',
            num_labels=self.num_classes,  # Platzhalter, wir überschreiben den Kopf sowieso
            ignore_mismatched_sizes=True
        )
        self.in_features = self.backbone.classifier.in_features

        self.dropout_rate = cfg.get('dropout_rate', 0.0)

        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.num_classes)
        )

        if self.freeze_backbone:
            for name, param in self.backbone.vit.named_parameters():
                param.requires_grad = False


    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits

    def overwrite_classifier(self):
        print("Overwriting classifier with custom architecture")
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.num_classes)
        )

        if self.freeze_backbone:
            for name, param in self.backbone.vit.named_parameters():
                param.requires_grad = False