from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.evaluation.operation_implementations.models.torch_rules import (
    TorchTabularFitPlan,
    build_torch_tabular_fit_plan,
)
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
        self.fit_plan: Optional[TorchTabularFitPlan] = None
        self.n_features_in_: Optional[int] = None
        self.feature_mean_: Optional[torch.Tensor] = None
        self.feature_scale_: Optional[torch.Tensor] = None
        self.target_mean_: Optional[torch.Tensor] = None
        self.target_scale_: Optional[torch.Tensor] = None

    def fit(self, input_data: TensorData) -> 'TorchTabularModel':
        features, target = self._fit_tensors(input_data)
        self.fit_plan = build_torch_tabular_fit_plan(
            params=self.params.to_dict(),
            samples_count=len(features),
            input_device=str(features.device),
            cuda_available=torch.cuda.is_available(),
            dataloader_kwargs=input_data.dataloader_kwargs,
        )
        self.device = torch.device(self.fit_plan.device)
        torch.manual_seed(self.fit_plan.random_state)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.fit_plan.random_state)

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

    def predict(self, input_data: TensorData) -> torch.Tensor:
        return self._prediction(input_data)

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
            self.fit_plan.random_state)
        indices = torch.randperm(
            samples_count, generator=generator).to(self.device)
        validation_count = int(
            round(samples_count * self.fit_plan.validation_fraction))
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
        widths = (n_features, *self.fit_plan.hidden_layer_sizes, output_width)
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
            lr=self.fit_plan.learning_rate,
            weight_decay=self.fit_plan.weight_decay,
        )
        best_loss = float('inf')
        best_state = copy.deepcopy(self.module.state_dict())
        patience_left = self.fit_plan.patience
        generator = torch.Generator(device=self.device).manual_seed(
            self.fit_plan.random_state)

        for _ in range(self.fit_plan.epochs):
            self.module.train()
            permutation = train_idx[torch.randperm(
                len(train_idx), generator=generator, device=self.device)]
            for batch_idx in permutation.split(self.fit_plan.batch_size):
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
                patience_left = self.fit_plan.patience
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
            for batch in features.split(self.fit_plan.batch_size)
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

    def predict_proba(self, input_data: TensorData) -> torch.Tensor:
        return self._prediction(input_data)

    def predict_labels(self, input_data: TensorData) -> torch.Tensor:
        probabilities = self.predict_proba(input_data)
        classes = self.classes_.to(probabilities.device)
        return classes[probabilities.argmax(dim=-1)]


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


class TorchMLPClassifier(TorchClassificationModel):
    """Feed-forward classifier trained natively on Torch tensors."""


class TorchLinearRegressor(TorchRegressionModel):
    """Linear regressor trained natively on Torch tensors."""

    def _build_module(self, n_features: int, output_width: int) -> nn.Module:
        return nn.Linear(n_features, output_width)

    def _train_validation_indices(self, samples_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.arange(samples_count, device=self.device)
        return indices, indices.new_empty(0)

    def _fit_module(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        train_idx: torch.Tensor,
        validation_idx: torch.Tensor,
    ):
        train_features = features[train_idx]
        design = torch.cat(
            [train_features, torch.ones(
                (len(train_features), 1), device=self.device)],
            dim=1,
        )
        coefficients = torch.linalg.lstsq(
            design, target[train_idx, None]).solution[:, 0]
        with torch.no_grad():
            self.module.weight.copy_(coefficients[:-1][None, :])
            self.module.bias.copy_(coefficients[-1:])


class TorchMLPRegressor(TorchRegressionModel):
    """Feed-forward regressor trained natively on Torch tensors."""
