from __future__ import annotations

from abc import abstractmethod
import copy
from dataclasses import replace
import random
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer
import torch
from torch import nn
from torch.nn import functional as F

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class TorchTabularModel(ModelImplementation):
    """Native Torch model contract for preprocessed tabular ``TensorData``.

    Required runtime fields are ``features`` (2D floating tensor), ``target`` for
    fit, ``task`` and ``dataloader_kwargs``. The fitted feature count and scaling
    statistics are retained so prediction follows exactly the fit-time contract.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.module: Optional[nn.Module] = None
        self.device = torch.device('cpu')
        self.n_features_in_: Optional[int] = None
        self.feature_mean_: Optional[torch.Tensor] = None
        self.feature_scale_: Optional[torch.Tensor] = None
        self.target_mean_: Optional[torch.Tensor] = None
        self.target_scale_: Optional[torch.Tensor] = None
        self.random_state = int(self.params.get('random_state', 42))

    def fit(self, input_data: TensorData) -> 'TorchTabularModel':
        features, target = self._fit_tensors(input_data)
        self._configure_training(input_data)
        torch.manual_seed(self.random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.random_state)

        features = features.to(self.device, dtype=torch.float32)
        target = target.to(self.device)
        train_idx, validation_idx = self._train_validation_indices(
            len(features))
        self._fit_scaler(features[train_idx])
        features = self._scale(features)
        target = self._prepare_target(target, train_idx)

        self.n_features_in_ = features.shape[1]
        self.module = self._build_module(
            self.n_features_in_, self._output_width()).to(self.device)
        self._fit_module(features, target, train_idx, validation_idx)
        self.module.eval()
        return self

    def predict(self, input_data: TensorData) -> TensorData:
        return replace(input_data, predict=self._prediction(input_data))

    def _configure_training(self, input_data: TensorData):
        params = self.params.to_dict()
        requested_batch_size = params.get('batch_size', input_data.dataloader_kwargs.get('batch_size', 32))
        self.batch_size = min(len(input_data.features), int(requested_batch_size))
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.validation_fraction = float(params.get('validation_fraction', 0.1))
        if not 0 <= self.validation_fraction < 1:
            raise ValueError('validation_fraction must be in [0, 1)')
        self.epochs = int(params.get('epochs', 200))
        self.patience = int(params.get('patience', 20))
        self.learning_rate = float(params.get('learning_rate', 0.01))
        if self.epochs <= 0:
            raise ValueError('epochs must be positive')
        if self.patience <= 0:
            raise ValueError('patience must be positive')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        self.device = torch.device(resolve_torch_device(
            requested_device=str(params.get('device', 'auto')),
            input_device=str(input_data.device),
            cuda_available=torch.cuda.is_available(),
        ))
        self.hidden_layer_sizes = normalize_hidden_layer_sizes(params.get('hidden_layer_sizes'))
        self.weight_decay = float(params.get('weight_decay', 0.0001))
        self.random_state = int(params.get('random_state', 42))

    def _fit_tensors(self, input_data: TensorData) -> tuple[torch.Tensor, torch.Tensor]:
        if input_data.target is None:
            raise ValueError(
                'TensorData.target is required to fit a Torch model')
        features = input_data.features
        if features.ndim != 2:
            raise ValueError(
                f'Torch tabular models require 2D features, got shape {tuple(features.shape)}')
        target = input_data.target.reshape(len(features), -1)
        if target.shape[1] != 1:
            raise ValueError(
                'Torch tabular models currently support a single target column')
        return features, target[:, 0]

    def _train_validation_indices(self, samples_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device='cpu').manual_seed(
            self.random_state)
        indices = torch.randperm(
            samples_count, generator=generator).to(self.device)
        validation_count = int(
            round(samples_count * self.validation_fraction))
        if validation_count == 0 or samples_count < 3:
            return indices, indices.new_empty(0)
        validation_count = min(validation_count, samples_count - 1)
        return indices[validation_count:], indices[:validation_count]

    def _fit_scaler(self, features: torch.Tensor):
        self.feature_mean_ = features.mean(dim=0)
        scale = features.std(dim=0, correction=0)
        self.feature_scale_ = torch.where(scale > torch.finfo(
            features.dtype).eps, scale, torch.ones_like(scale))

    def _scale(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean_) / self.feature_scale_

    def _build_module(self, n_features: int, output_width: int) -> nn.Module:
        widths = (n_features, *self.hidden_layer_sizes, output_width)
        layers = []
        for layer_idx, (input_width, next_width) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(input_width, next_width))
            if layer_idx < len(widths) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _fit_module(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        train_idx: torch.Tensor,
        validation_idx: torch.Tensor,
    ):
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        best_loss = float('inf')
        best_state = copy.deepcopy(self.module.state_dict())
        patience_left = self.patience
        generator = torch.Generator(device=self.device).manual_seed(
            self.random_state)

        for _ in range(self.epochs):
            self.module.train()
            permutation = train_idx[torch.randperm(
                len(train_idx), generator=generator, device=self.device)]
            for batch_idx in permutation.split(self.batch_size):
                optimizer.zero_grad()
                loss = self._loss(self.module(
                    features[batch_idx]), target[batch_idx])
                loss.backward()
                optimizer.step()

            if len(validation_idx) == 0:
                continue
            validation_loss = self._validation_loss(
                features[validation_idx], target[validation_idx])
            if validation_loss < best_loss:
                best_loss = validation_loss
                patience_left = self.patience
                best_state = copy.deepcopy(self.module.state_dict())
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        if len(validation_idx) > 0:
            self.module.load_state_dict(best_state)

    @torch.inference_mode()
    def _validation_loss(self, features: torch.Tensor, target: torch.Tensor) -> float:
        self.module.eval()
        return float(self._loss(self.module(features), target).detach().cpu())

    @torch.inference_mode()
    def _prediction(self, input_data: TensorData) -> torch.Tensor:
        if self.module is None:
            raise ValueError(f'{self.__class__.__name__} is not fitted yet')
        features = input_data.features
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f'Expected 2D features with {self.n_features_in_} columns, got {tuple(features.shape)}')
        source_device = features.device
        features = self._scale(features.to(self.device, dtype=torch.float32))
        self.module.eval()
        predictions = [
            self.module(batch)
            for batch in features.split(self.batch_size)
        ]
        return self._postprocess_prediction(torch.cat(predictions)).to(source_device)

    @abstractmethod
    def _prepare_target(self, target: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _output_width(self) -> int:
        pass

    @abstractmethod
    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _postprocess_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        pass


class TorchClassificationModel(TorchTabularModel):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_: Optional[torch.Tensor] = None

    def _prepare_target(self, target: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
        self.classes_, encoded = torch.unique(
            target, sorted=True, return_inverse=True)
        return encoded.long()

    def _output_width(self) -> int:
        return len(self.classes_)

    @staticmethod
    def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(prediction, target)

    @staticmethod
    def _postprocess_prediction(prediction: torch.Tensor) -> torch.Tensor:
        return torch.softmax(prediction, dim=-1)

    def predict_proba(self, input_data: TensorData) -> TensorData:
        return replace(input_data, predict=self._prediction(input_data))

    def predict(self, input_data: TensorData) -> TensorData:
        probabilities = self._prediction(input_data)
        classes = self.classes_.to(probabilities.device)
        return replace(input_data, predict=classes[probabilities.argmax(dim=-1)])


class TorchRegressionModel(TorchTabularModel):
    def _prepare_target(self, target: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=torch.float32)
        self.target_mean_ = target[train_idx].mean()
        scale = target[train_idx].std(correction=0)
        self.target_scale_ = torch.where(
            scale > torch.finfo(target.dtype).eps,
            scale,
            torch.ones_like(scale),
        )
        return (target - self.target_mean_) / self.target_scale_

    @staticmethod
    def _output_width() -> int:
        return 1

    @staticmethod
    def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(prediction[:, 0], target)

    def _postprocess_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction[:, 0] * self.target_scale_ + self.target_mean_


class TorchLinearClassifier(TorchClassificationModel):
    """Multinomial linear classifier trained natively on Torch tensors."""

    def _build_module(self, n_features: int, output_width: int) -> nn.Module:
        return nn.Linear(n_features, output_width)








def resolve_torch_device(requested_device: str, input_device: str, cuda_available: bool) -> str:
    """Resolve model device while preserving the TensorData runtime by default."""
    if requested_device == 'auto':
        return input_device
    if requested_device.startswith('cuda') and not cuda_available:
        raise ValueError("device='cuda' was requested, but CUDA is unavailable")
    if requested_device == 'cuda' and input_device.startswith('cuda:'):
        return input_device
    if requested_device != 'cpu' and requested_device != 'cuda' and not requested_device.startswith('cuda:'):
        raise ValueError(f'Unsupported Torch device: {requested_device!r}')
    return requested_device


def normalize_hidden_layer_sizes(value: Any) -> tuple[int, ...]:
    """Normalize sklearn-compatible hidden-layer notation."""
    if value is None:
        return ()
    if isinstance(value, int):
        sizes = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sizes = tuple(int(size) for size in value)
    else:
        raise ValueError('hidden_layer_sizes must be an int or a sequence of ints')
    if any(size <= 0 for size in sizes):
        raise ValueError('hidden_layer_sizes values must be positive')
    return sizes
