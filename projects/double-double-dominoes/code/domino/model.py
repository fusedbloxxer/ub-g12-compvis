import os
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import typing as t
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchmetrics
import torchvision as tv
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from torchvision.models import resnet50, ResNet50_Weights
from torch import Tensor
import torch.utils.data as data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import torchmetrics as tm
import torchmetrics.classification
from torchmetrics.classification import MulticlassConfusionMatrix
from imblearn.ensemble import BalancedRandomForestClassifier
import pathlib as pb
import cv2 as cv
import pickle, joblib
import abc

from .util import same_cell_dim
from .preprocess import DataAugmentation, DataPreprocess


class ResNetCellClassifier(pl.LightningModule):
    def __init__(self, *args: Any, full_finetune: bool = False, augment: bool=True, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Consider augmenting the data
        self.augment = DataAugmentation() if augment else nn.Identity()

        # Load pretrained model
        self.pretrained_weights = ResNet50_Weights.IMAGENET1K_V2
        self.pretrained_transforms: nn.Module = self.pretrained_weights.transforms(antialias=True)
        pretrained_model: tv.models.ResNet = resnet50(weights=self.pretrained_weights)
        self.num_filters: int = pretrained_model.fc.in_features

        # Select all layers but the last
        self.n_classes: int = 8
        self.pretrained_layers = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.trainable_layers = nn.Linear(in_features=self.num_filters, out_features=self.n_classes)
        self.pretrained_layers.requires_grad_(full_finetune)

        # Define the training objective
        self.train_conf_matrix = MulticlassConfusionMatrix(self.n_classes)
        self.valid_conf_matrix = MulticlassConfusionMatrix(self.n_classes)
        self.test_conf_matrix = MulticlassConfusionMatrix(self.n_classes)
        self.train_balanced_accuracy = BalancedAccuracy()
        self.valid_balanced_accuracy = BalancedAccuracy()
        self.test_balanced_accuracy = BalancedAccuracy()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        features: Tensor = self.pretrained_transforms.forward(x)
        features: Tensor = self.pretrained_layers.forward(features).flatten(1)
        output: Tensor = self.trainable_layers.forward(features)
        return output

    def predict_step(self, batch: t.Tuple[Tensor, Tensor] | Tensor, batch_idx: int) -> Any:
        return self(batch[0])

    def training_step(self, batch: t.Tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        self.pretrained_layers.eval(); self.trainable_layers.train()
        x, y_true = batch

        # Augment the images solely during training
        with torch.no_grad():
            features: Tensor = self.augment.forward(x)

        # Predict on the input data
        y_hat: Tensor = self.forward(features)
        loss: Tensor = self.loss_fn.forward(y_hat, y_true)

        # Compute metrics
        with torch.no_grad():
            self.train_balanced_accuracy.update(y_hat.argmax(dim=1), y_true)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_accuracy_epoch', self.train_balanced_accuracy.compute(), prog_bar=True)
        self.train_balanced_accuracy.reset()

    def validation_step(self, batch: t.Tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT | None:
        x, y_true = batch

        with torch.no_grad():
            y_hat: Tensor = self.forward(x)
            loss: Tensor = self.loss_fn.forward(y_hat, y_true)

            self.valid_conf_matrix.update(y_hat, y_true)
            self.valid_balanced_accuracy.update(y_hat.argmax(dim=1), y_true)
            self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log('valid_accuracy_epoch', self.valid_balanced_accuracy.compute(), prog_bar=True)
        ConfusionMatrixDisplay(self.valid_conf_matrix.compute().cpu().numpy()).plot()
        self.valid_balanced_accuracy.reset()
        self.valid_conf_matrix.reset()

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT | None:
        x, y_true = batch

        with torch.no_grad():
            y_hat: Tensor = self.forward(x)
            self.test_conf_matrix.update(y_hat, y_true)
            self.test_balanced_accuracy.update(y_hat.argmax(dim=1), y_true)

    def on_test_epoch_end(self) -> None:
        self.log('test_accuracy_epoch', self.test_balanced_accuracy.compute(), prog_bar=True)
        ConfusionMatrixDisplay(self.test_conf_matrix.compute().cpu().numpy()).plot()
        self.test_balanced_accuracy.reset()
        self.test_conf_matrix.reset()

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.trainable_layers.parameters(), lr=1e-4)


class CellClassifier(abc.ABC):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def __call__(self, grid_cells: t.List[t.List[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def load_checkpoint(path: pb.Path, *args, **kwargs) -> 'CellClassifier':
        raise NotImplementedError()


class RandomForestCellClassifier(CellClassifier):
    def __init__(self, model: BalancedRandomForestClassifier, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model: BalancedRandomForestClassifier = model
        self.train_mean: float = 100.0261724005487
        self.train_std: float = 42.46173079733813

    def __call__(self, grid_cells: t.List[t.List[np.ndarray]]) -> np.ndarray:
        # Ensure all sizes match
        cells_grayscale: np.ndarray = same_cell_dim(grid_cells)

        # Flatten the image pixels
        cells_features: np.ndarray = cells_grayscale.reshape(cells_grayscale.shape[0], -1)

        # Normalize the data
        cells_norm: np.ndarray = (cells_features - self.train_mean) / self.train_std

        # Predict what the cells represent
        outputs: np.ndarray = self.model.predict(X=cells_norm)
        return outputs.reshape((len(grid_cells), len(grid_cells[0])))

    @staticmethod
    def load_checkpoint(path: pb.Path) -> 'RandomForestCellClassifier':
        return RandomForestCellClassifier(model=joblib.load(path))


class PretrainedCellClassifier(CellClassifier):
    def __init__(self, model: ResNetCellClassifier, num_workers: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainer = pl.Trainer(default_root_dir=pb.Path('..', '.model'))
        self.preprocess: nn.Module = DataPreprocess()
        self.model: ResNetCellClassifier = model
        self.num_workers: int = num_workers

    def __call__(self, grid_cells: t.List[t.List[np.ndarray]]) -> np.ndarray:
        # Ensure all sizes match
        cells_grayscale: np.ndarray = same_cell_dim(grid_cells)

        # Grayscale -> RGB and [0, 255] -> [0., 1.]
        cells_rgb: Tensor = self.preprocess.forward(cells_grayscale)

        # Avoid OutOfMemory GPU errors by using a dataset
        cells_dataset: data.Dataset   = data.TensorDataset(cells_rgb)
        cells_loader: data.DataLoader = data.DataLoader(cells_dataset, batch_size=8, num_workers=self.num_workers)

        # Infer what the cells represent
        outputs: t.List[t.Any] | None = self.trainer.predict(self.model, dataloaders=cells_loader)
        assert outputs is not None, 'Fatal error... prediction returned None!'

        # Transform to labels
        y_pred: Tensor = torch.cat(outputs, dim=0).argmax(dim=1)
        return y_pred.numpy().reshape((len(grid_cells), len(grid_cells[0]))).astype(np.int32)

    @staticmethod
    def load_checkpoint(path: pb.Path, *args, **kwargs) -> 'PretrainedCellClassifier':
        return PretrainedCellClassifier(ResNetCellClassifier.load_from_checkpoint(path), *args, **kwargs)


class BalancedAccuracy(tm.Metric):
    y_true: t.List[Tensor]
    y_pred: t.List[Tensor]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.add_state('y_true', default=[], dist_reduce_fx=None)
        self.add_state('y_pred', default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert target.shape == preds.shape
        self.y_pred.append(preds.detach().cpu())
        self.y_true.append(target.detach().cpu())

    def compute(self) -> float:
        y_true: np.ndarray = torch.cat(self.y_true, dim=0).numpy()
        y_pred: np.ndarray = torch.cat(self.y_pred, dim=0).numpy()
        return balanced_accuracy_score(y_true, y_pred)

