import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from abc import ABC, abstractmethod
from typing import final
from torchmetrics import MetricCollection
import torchmetrics
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

class LitBaseModel(L.LightningModule, ABC):
    """
    An abstract base class for PyTorch Lightning models to standardize training.

    This class provides a non-overridable (`@final`) training, validation, and
    testing framework. It automatically handles metric calculation for a
    5-class classification task, loss computation, logging, and optimizer setup.
    Subclasses are only required to implement their own architecture in `__init__`
    and the `forward` pass.

    The configuration for the optimizer and other model parameters is expected
    to be passed in a `params` dictionary.

    Attributes:
        params (dict): A dictionary containing model and optimizer configurations.
        train_confmat (torchmetrics.ConfusionMatrix): Metric to compute the confusion matrix for training data.
        val_confmat (torchmetrics.ConfusionMatrix): Metric to compute the confusion matrix for validation data.
        test_confmat (torchmetrics.ConfusionMatrix): Metric to compute the confusion matrix for test data.

    Example:
        >>> class SimpleCNN(LitBaseModel):
        ...     def __init__(self, params):
        ...         super().__init__(params)
        ...         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        ...         self.fc1 = nn.LazyLinear(5) # Example dimensions
        ...
        ...     def forward(self, x):
        ...         x = F.relu(self.conv1(x))
        ...         x = torch.flatten(x, 1)
        ...         return self.fc1(x)
        ...
        >>> # Training would be handled by a PyTorch Lightning Trainer.
    """
    def __init__(self, params: dict):
        """
        Initializes the LitBaseModel.

        Args:
            params (dict): A dictionary containing hyperparameters, typically loaded
                           from a YAML file. It must include model and optimizer settings.
                           Example: {'model': {'optimizer': 'Adam', 'optimizer_params': {'lr': 0.001}}}
        """
        super().__init__()
        self.save_hyperparameters("params")
        self.params = params
        self.training_step_outputs = []
        self.validation_step_outputs = []

        warnings.filterwarnings(
            "ignore",
            message="No positive samples in targets, true positive.*",
            category=UserWarning,
            module="torchmetrics.utilities.prints",
        )

        self.num_classes = params['model']['model_params']['num_classes']
        self.train_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.train_metrics = MetricCollection({
            'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='micro'),
            'precision_macro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes,
                                                         average='weighted'),
            'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='micro'),
            'recall_macro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'accuracy_micro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro'),
            'accuracy_macro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            'accuracy_weighted': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes,
                                                       average='weighted'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro'),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
        }, prefix="train_")

        self.val_metrics = MetricCollection({
            'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='micro'),
            'precision_macro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes,
                                                         average='weighted'),
            'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='micro'),
            'recall_macro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'accuracy_micro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro'),
            'accuracy_macro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            'accuracy_weighted': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes,
                                                       average='weighted'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro'),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
        }, prefix="val_")


    @final
    def training_step(self, batch: tuple, batch_idx: tuple) -> torch.Tensor:
        """
        Performs a single training step. This method is final and should not be overridden.

        Computes the loss and logs a collection of metrics (Accuracy, Precision, Recall, F1, AUROC).

        Args:
            batch (tuple): A tuple containing the input tensor and the ground truth labels (x, y).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The cross-entropy loss for the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 2) Update stateful metrics (no logging here)
        self.train_metrics.update(logits, y)
        self.train_confmat.update(logits, y)

        # Still log loss on‐step if you like
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @final
    def on_train_epoch_end(self):
        """
        Hook called at the end of the training epoch. This method is final.

        Computes, plots, and logs the training confusion matrix to the logger (WandB).
        """
        metrics = self.train_metrics.compute()
        cm = self.train_confmat.compute().cpu().numpy()

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Train Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        self.logger.experiment.log({"train_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        self.train_metrics.reset()
        self.train_confmat.reset()


    @final
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single validation step. This method is final and should not be overridden.

        Computes the loss and logs a collection of metrics (Accuracy, Precision, Recall, F1, AUROC).

        Args:
            batch (tuple): A tuple containing the input tensor and the ground truth labels (x, y).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The cross-entropy loss for the current validation batch.
        """

        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 2) Update stateful metrics (no logging here)
        self.val_metrics.update(logits, y)
        self.val_confmat.update(logits, y)

        # Still log loss on‐step if you like
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @final
    def on_validation_epoch_end(self):
        """
        Hook called at the end of the validation epoch. This method is final.

        Computes, plots, and logs the validation confusion matrix to the logger (WandB).
        """
        metrics = self.val_metrics.compute()
        cm = self.val_confmat.compute().cpu().numpy()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        # plot it
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Validation Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # send to W&B
        self.logger.experiment.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        self.val_metrics.reset()
        self.val_confmat.reset()

    @final
    def test_step(self, batch: tuple, batch_idx: int):
        """
        Performs a single test step. This method is final and should not be overridden.

        Computes the loss and logs a collection of metrics (Accuracy, Precision, Recall, F1, AUROC).

        Args:
            batch (tuple): A tuple containing the input tensor and the ground truth labels (x, y).
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        metrics = MetricCollection({
            'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='micro'),
            'precision_macro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='micro'),
            'recall_macro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'accuracy_micro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro'),
            'accuracy_macro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            'accuracy_weighted': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro'),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'precision_recall_curve_macro': torchmetrics.AUROC(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_recall_curve_weighted': torchmetrics.AUROC(task='multiclass', num_classes=self.num_classes, average='weighted'),
        }, prefix="test")

        metrics = metrics.to(self.device)
        dict = metrics(y_hat, y)
        dict['test_cross_entropy'] = loss

        self.test_confmat.update(y_hat, y)

        self.log_dict(dict, on_step=True, on_epoch=True, prog_bar=True)

    @final
    def on_test_epoch_end(self):
        """
        Hook called at the end of the test epoch. This method is final.

        Computes, plots, and logs the test confusion matrix to the logger (WandB).
        """
        cm = self.test_confmat.compute().cpu().numpy()
        self.test_confmat.reset()

        # plot it
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # send to W&B
        self.logger.experiment.log({"test_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)



    @final
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model. This method is final.

        It reads the optimizer name ('Adam' or 'SGD') and its parameters from the
        `self.params` dictionary provided during initialization.

        Raises:
            ValueError: If the optimizer name in the params is not 'Adam' or 'SGD'.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer_name = self.params['model']['optimizer']
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.params['model']['optimizer_params'])
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.params['model']['optimizer_params'])
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized.")
        return optimizer


