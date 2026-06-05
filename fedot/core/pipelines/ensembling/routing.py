from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.multi_modal import MultiModalData


@dataclass(frozen=True)
class SamplingRoutingContext:
    predictor: Any
    partition_names: Sequence[str]

    @classmethod
    def from_predictor(cls,
                       predictor: Any,
                       partition_names: Sequence[str]) -> 'SamplingRoutingContext':
        return cls(
            predictor=predictor,
            partition_names=tuple(str(name) for name in partition_names),
        )

    @property
    def can_predict_partitions(self) -> bool:
        return callable(getattr(self.predictor, 'predict_partitions', None))

    @property
    def can_predict_partition_proba(self) -> bool:
        return callable(getattr(self.predictor, 'predict_partition_proba', None))

    @property
    def can_transform_embedding(self) -> bool:
        return callable(getattr(self.predictor, 'transform_embedding', None))

    def base_weights(self, input_data: InputData | MultiModalData, active_names: Sequence[str]) -> np.ndarray:
        if not isinstance(input_data, InputData):
            raise ValueError('Sampling routing supports only InputData.')
        features = input_data.to_dataframe()
        if self.can_predict_partition_proba:
            proba = np.asarray(self.predictor.predict_partition_proba(features), dtype=float)
            return self._align_proba(proba, active_names)
        if self.can_predict_partitions:
            labels = np.asarray(self.predictor.predict_partitions(features))
            return self._align_labels(labels, active_names)
        raise ValueError('Sampling predictor has no partition prediction method.')

    def transform_features(self, input_data: InputData | MultiModalData) -> np.ndarray:
        if not isinstance(input_data, InputData):
            raise ValueError('Sampling routing supports only InputData.')
        features = input_data.to_dataframe()
        if self.can_transform_embedding:
            return np.asarray(self.predictor.transform_embedding(features), dtype=np.float32)

        numeric = features.select_dtypes(include=[np.number, 'bool'])
        if numeric.shape[1] == 0:
            raise ValueError('Gated ensemble requires routing embeddings or numeric sampling features.')
        return numeric.to_numpy(dtype=np.float32)

    def _align_proba(self, proba: np.ndarray, active_names: Sequence[str]) -> np.ndarray:
        if proba.ndim != 2:
            raise ValueError('Partition probabilities must be a 2D matrix.')
        if proba.shape[1] != len(self.partition_names):
            raise ValueError('Partition probability columns do not match partition names.')

        name_to_col = {str(name): idx for idx, name in enumerate(self.partition_names)}
        aligned = np.zeros((proba.shape[0], len(active_names)), dtype=float)
        for model_idx, name in enumerate(active_names):
            if str(name) in name_to_col:
                aligned[:, model_idx] = proba[:, name_to_col[str(name)]]
        return _normalize_rows(aligned)

    def _align_labels(self, labels: np.ndarray, active_names: Sequence[str]) -> np.ndarray:
        labels = labels.reshape(-1)
        aligned = np.zeros((labels.shape[0], len(active_names)), dtype=float)
        for model_idx, name in enumerate(active_names):
            aligned[:, model_idx] = self._labels_match_name(labels, str(name)).astype(float)
        return _normalize_rows(aligned)

    @staticmethod
    def _labels_match_name(labels: np.ndarray, name: str) -> np.ndarray:
        if np.issubdtype(labels.dtype, np.number):
            suffix = _numeric_suffix(name)
            if suffix is not None:
                return labels.astype(int) == suffix
        return labels.astype(str) == name


class ConstrainedGatingRouter:
    def __init__(self,
                 hidden_dim: int = 64,
                 epochs: int = 200,
                 lr: float = 1e-3,
                 kl_weight: float = 0.10,
                 balance_weight: float = 0.01,
                 weight_decay: float = 1e-4,
                 batch_size: int = 2048,
                 device: str = 'auto'):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.balance_weight = balance_weight
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.device_ = None
        self.input_mean_ = None
        self.input_scale_ = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(self,
            features: np.ndarray,
            prior_weights: np.ndarray,
            predictions: np.ndarray,
            target: np.ndarray) -> 'ConstrainedGatingRouter':
        features = np.asarray(features, dtype=np.float32)
        prior_weights = np.asarray(prior_weights, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32).reshape(-1)

        self.input_mean_ = features.mean(axis=0)
        self.input_scale_ = features.std(axis=0) + 1e-6
        x_np = self._standardize(features)

        device = self._resolve_device()
        model = GatingNetwork(
            n_features=x_np.shape[1],
            n_models=predictions.shape[1],
            hidden_dim=max(4, int(self.hidden_dim)),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        prior = torch.as_tensor(prior_weights, dtype=torch.float32, device=device)
        pred = torch.as_tensor(predictions, dtype=torch.float32, device=device)
        y = torch.as_tensor(target, dtype=torch.float32, device=device)

        n_samples = x.shape[0]
        batch_size = max(1, min(int(self.batch_size), int(n_samples)))
        for _ in range(max(1, int(self.epochs))):
            order = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                weights = torch.softmax(model(x[idx]), dim=1)
                y_hat = torch.sum(weights * pred[idx], dim=1)
                mse = torch.mean((y_hat - y[idx]) ** 2)
                kl = torch.mean(torch.sum(weights * (torch.log(weights + 1e-8) - torch.log(prior[idx] + 1e-8)), dim=1))
                uniform = torch.full_like(weights.mean(dim=0), 1.0 / weights.shape[1])
                balance = torch.mean((weights.mean(dim=0) - uniform) ** 2)
                loss = mse + float(self.kl_weight) * kl + float(self.balance_weight) * balance
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = model.eval()
        self.device_ = device
        return self

    def weights(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError('Constrained gating router is not fitted.')

        x_np = self._standardize(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            logits = self.model(torch.as_tensor(x_np, dtype=torch.float32, device=self.device_))
            return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.input_mean_) / self.input_scale_

    def _resolve_device(self) -> torch.device:
        requested = str(self.device).strip().lower()
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda' and not torch.cuda.is_available():
            requested = 'cpu'
        return torch.device(requested)


def _normalize_rows(weights: np.ndarray) -> np.ndarray:
    row_sums = weights.sum(axis=1, keepdims=True)
    fallback = np.full_like(weights, 1.0 / max(weights.shape[1], 1), dtype=float)
    return np.divide(weights, row_sums, out=fallback, where=row_sums > 0)


def _numeric_suffix(name: str) -> Optional[int]:
    value = name.rsplit('_', 1)[-1]
    return int(value) if value.isdigit() else None


class GatingNetwork:
    def __new__(cls, n_features: int, n_models: int, hidden_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_models),
        )
