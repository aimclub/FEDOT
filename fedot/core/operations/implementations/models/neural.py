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

from fedot.core.backend.array_conversions import as_numpy, as_torch
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


class NeuralTabularModelImplementation(ModelImplementation):
    """Base implementation for experimental TensorData-only tabular models."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None
        self.classes_ = None
        self.cat_col_names = []
        self.feature_names = None

    def _to_frame(self, tensor_data: TensorData) -> pd.DataFrame:
        features = as_numpy(tensor_data.features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        feature_names = as_numpy(tensor_data.features_names)
        if feature_names is None or len(feature_names) != features.shape[1]:
            feature_names = np.array(
                [f'f_{i}' for i in range(features.shape[1])])
        feature_names = [str(name) for name in feature_names]
        self.feature_names = feature_names

        frame = pd.DataFrame(features, columns=feature_names)
        categorical_idx = as_numpy(tensor_data.categorical_idx)
        if categorical_idx is None:
            categorical_idx = []
        self.cat_col_names = [feature_names[int(
            idx)] for idx in categorical_idx if int(idx) < len(feature_names)]
        for col in self.cat_col_names:
            frame[col] = frame[col].astype('category')
        return frame

    def _target(self, tensor_data: TensorData):
        target = as_numpy(tensor_data.target)
        if target is not None and target.ndim > 1 and target.shape[1] == 1:
            target = target.ravel()
        return target

    def _device(self, tensor_data: TensorData) -> str:
        device = self.params.get('device', 'auto')
        if device != 'auto':
            return device
        if tensor_data.device.type == 'cuda':
            return str(tensor_data.device)
        return 'cpu'

    def _output(self, tensor_data: TensorData, prediction: np.ndarray) -> TensorData:
        prediction = as_torch(
            prediction,
            device=tensor_data.features.device,
        )
        return replace(tensor_data, predict=prediction)

    def _encode_target(self, y):
        if self._is_classification:
            self.classes_, encoded = np.unique(y, return_inverse=True)
            return encoded.astype(np.int64)
        return y.astype(np.float32)

    @property
    def _is_classification(self) -> bool:
        return False

    def _decode_prediction(self, prediction):
        if self._is_classification and self.classes_ is not None:
            return self.classes_[np.asarray(prediction, dtype=int)]
        return prediction


class NeuralTabularClassificationImplementation(NeuralTabularModelImplementation):
    @property
    def _is_classification(self) -> bool:
        return True


class TabMCategoryEncoder:
    def __init__(self):
        self.mappings = []
        self.cardinalities = []

    def fit(self, values: np.ndarray):
        self.mappings = []
        self.cardinalities = []
        for col_idx in range(values.shape[1]):
            col = values[:, col_idx]
            col = col[~pd.isna(col)]
            categories = np.unique(col)
            mapping = {category: idx for idx,
                       category in enumerate(categories.tolist())}
            self.mappings.append(mapping)
            self.cardinalities.append(len(mapping))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        encoded = np.empty(values.shape, dtype=np.int64)
        for col_idx, mapping in enumerate(self.mappings):
            unknown_idx = len(mapping)
            encoded[:, col_idx] = [
                mapping.get(value, unknown_idx) if not pd.isna(
                    value) else unknown_idx
                for value in values[:, col_idx]
            ]
        return encoded


class TabMPreprocessor:
    def __init__(self, categorical_idx: np.ndarray, random_state: int, num_emb_n_bins: int):
        self.categorical_idx = np.asarray(categorical_idx, dtype=int)
        self.numerical_idx = None
        self.cat_encoder = TabMCategoryEncoder()
        self.num_imputer = None
        self.num_quantile_transformer = None
        self.num_bins_encoder = None
        self.num_col_mask = None
        self.random_state = random_state
        self.num_emb_n_bins = num_emb_n_bins

    def fit(self, features: np.ndarray):
        all_idx = np.arange(features.shape[1])
        self.categorical_idx = self.categorical_idx[self.categorical_idx <
                                                    features.shape[1]]
        self.numerical_idx = np.setdiff1d(all_idx, self.categorical_idx)

        if len(self.categorical_idx) > 0:
            self.cat_encoder.fit(features[:, self.categorical_idx])
        if len(self.numerical_idx) > 0:
            numerical = features[:, self.numerical_idx].astype(np.float32)
            n_quantiles = min(max(len(numerical) // 2, 10),
                              1000, len(numerical))
            self.num_imputer = SimpleImputer(add_indicator=True)
            self.num_quantile_transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution='normal',
                random_state=self.random_state,
            )
            numerical = self.num_imputer.fit_transform(numerical)
            quantile_features = self.num_quantile_transformer.fit_transform(
                numerical)
            if self.num_emb_n_bins > 1:
                self.num_bins_encoder = KBinsDiscretizer(
                    n_bins=min(self.num_emb_n_bins, max(
                        2, len(numerical) // 2)),
                    encode='onehot-dense',
                    strategy='quantile',
                    random_state=self.random_state,
                )
                binned_features = self.num_bins_encoder.fit_transform(
                    numerical)
                numerical = np.hstack([quantile_features, binned_features])
            else:
                numerical = quantile_features
            self.num_col_mask = np.nanstd(numerical, axis=0) > 1e-12
        else:
            self.num_col_mask = np.array([], dtype=bool)
        return self

    def transform(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(self.numerical_idx) > 0:
            numerical = features[:, self.numerical_idx].astype(np.float32)
            numerical = self.num_imputer.transform(numerical)
            quantile_features = self.num_quantile_transformer.transform(
                numerical)
            if self.num_bins_encoder is not None:
                binned_features = self.num_bins_encoder.transform(numerical)
                numerical = np.hstack([quantile_features, binned_features])
            else:
                numerical = quantile_features
            numerical = numerical[:, self.num_col_mask].astype(np.float32)
        else:
            numerical = np.empty((len(features), 0), dtype=np.float32)

        if len(self.categorical_idx) > 0:
            categorical = self.cat_encoder.transform(
                features[:, self.categorical_idx])
        else:
            categorical = np.empty((len(features), 0), dtype=np.int64)

        return numerical, categorical


class TabMNetwork(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        output_dim: int,
        d_block: int,
        dropout: float,
        tabm_k: int,
        n_blocks: int,
        arch_type: str,
        share_training_batches: bool,
    ):
        super().__init__()
        import tabm as tabm_lib

        self.k = tabm_k
        self.share_training_batches = share_training_batches
        self.model = tabm_lib.TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=[cardinality +
                               1 for cardinality in cat_cardinalities] or None,
            d_out=output_dim,
            d_block=d_block,
            dropout=dropout,
            k=tabm_k,
            n_blocks=n_blocks,
            arch_type=arch_type,
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.training and not self.share_training_batches:
            x_num = x_num.reshape(len(x_num) // self.k,
                                  self.k, *x_num.shape[1:])
            x_cat = x_cat.reshape(len(x_cat) // self.k,
                                  self.k, *x_cat.shape[1:])
        x_num = x_num if x_num.shape[-1] > 0 else None
        x_cat = x_cat if x_cat.shape[-1] > 0 else None
        return self.model(x_num, x_cat)


def make_tabm_parameter_groups(model: nn.Module) -> list[dict]:
    zero_weight_decay_params = []
    default_params = []
    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name.endswith('bias'):
                zero_weight_decay_params.append(parameter)
            else:
                default_params.append(parameter)
    return [
        {'params': default_params},
        {'params': zero_weight_decay_params, 'weight_decay': 0.0},
    ]


class TabMImplementation(NeuralTabularModelImplementation):
    def fit(self, input_data: TensorData):
        self._setup_random_state()
        X = as_numpy(input_data.features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = self._encode_target(self._target(input_data))

        X_train, X_val, y_train, y_val = self._train_val_split(X, y)
        categorical_idx = as_numpy(input_data.categorical_idx)
        if categorical_idx is None:
            categorical_idx = np.array([], dtype=int)
        self.preprocessor = TabMPreprocessor(
            categorical_idx=categorical_idx,
            random_state=self.params.get('random_state', 42),
            num_emb_n_bins=self.params.get(
                'num_emb_n_bins', self._default_num_emb_n_bins()),
        ).fit(X_train)

        x_num_train, x_cat_train = self.preprocessor.transform(X_train)
        x_num_val, x_cat_val = self.preprocessor.transform(X_val)
        train_tensors = self._to_torch_tensors(
            x_num_train, x_cat_train, y_train, fit_regression_target=True)
        val_tensors = self._to_torch_tensors(x_num_val, x_cat_val, y_val)

        arch_type = self.params.get('arch_type', 'tabm-mini')
        self.model = self._make_model(
            x_num_train.shape[1], arch_type).to(self.device_)
        self._fit_model(train_tensors, val_tensors)
        return self

    def _make_model(self, n_num_features: int, arch_type: str) -> nn.Module:
        return TabMNetwork(
            n_num_features=n_num_features,
            cat_cardinalities=self.preprocessor.cat_encoder.cardinalities,
            output_dim=self._output_dim(),
            d_block=self.params.get('d_block', 256),
            dropout=self.params.get('dropout', 0.1),
            tabm_k=self.params.get('tabm_k', 8),
            n_blocks=self.params.get('n_blocks', 2),
            arch_type=arch_type,
            share_training_batches=self.params.get(
                'share_training_batches', False),
        )

    def _setup_random_state(self):
        seed = self.params.get('random_state', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_num_threads(self.params.get('n_jobs', 1))
        self.device_ = torch.device(self._device_from_params())

    def _device_from_params(self) -> str:
        device = self.params.get('device', 'auto')
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _output_dim(self) -> int:
        if self._is_classification:
            return len(self.classes_)
        return 1

    def _default_num_emb_n_bins(self) -> int:
        return 0 if self._is_classification else 16

    def _to_torch_tensors(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        fit_regression_target: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self._is_classification:
            y_tensor = as_torch(
                y, dtype=torch.long, device=self.device_)
        else:
            y = y.astype(np.float32)
            if fit_regression_target:
                self.y_mean_ = float(np.mean(y))
                self.y_std_ = float(np.std(y))
            y = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            y_tensor = as_torch(
                y, dtype=torch.float32, device=self.device_)

        return {
            'x_num': as_torch(x_num, dtype=torch.float32, device=self.device_),
            'x_cat': as_torch(x_cat, dtype=torch.long, device=self.device_),
            'y': y_tensor,
        }

    def _fit_model(self, train_tensors: dict[str, torch.Tensor], val_tensors: dict[str, torch.Tensor]):
        optimizer = torch.optim.AdamW(
            make_tabm_parameter_groups(self.model),
            lr=self.params.get('lr', 0.002),
            weight_decay=self.params.get('weight_decay', 0.0003),
        )
        n_train = len(train_tensors['y'])
        batch_size = self._batch_size(n_train)
        best_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        remaining_patience = self.params.get('patience', 6)

        for _ in range(self.params.get('n_epochs', 30)):
            self.model.train()
            for batch_idx in self._batch_indices(n_train, batch_size):
                optimizer.zero_grad()
                prediction = self.model(
                    train_tensors['x_num'][batch_idx], train_tensors['x_cat'][batch_idx])
                loss = self._loss(prediction, train_tensors['y'][batch_idx])
                loss.backward()
                gradient_norm = self.params.get('gradient_clipping_norm', 1.0)
                if gradient_norm is not None and gradient_norm != 'none':
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_norm)
                optimizer.step()

            val_loss = self._validation_loss(val_tensors)
            if val_loss < best_loss:
                best_loss = val_loss
                remaining_patience = self.params.get('patience', 6)
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                remaining_patience -= 1
                if remaining_patience < 0:
                    break

        self.model.load_state_dict(best_state)

    def _batch_indices(self, n_train: int, batch_size: int):
        if self.model.share_training_batches:
            return torch.randperm(n_train, device=self.device_).split(batch_size)
        member_permutations = torch.rand(
            (self.model.k, n_train), device=self.device_).argsort(dim=1)
        return [
            batch.transpose(0, 1).flatten()
            for batch in member_permutations.split(batch_size, dim=1)
        ]

    def _batch_size(self, n_train: int) -> int:
        batch_size = self.params.get('batch_size', 'auto')
        if batch_size == 'auto':
            return min(256, n_train)
        return min(int(batch_size), n_train)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_prediction_size = prediction.flatten(0, 1).shape[0]
        if len(target) != flat_prediction_size:
            k = prediction.shape[1]
            target = target.repeat_interleave(k)
        if self._is_classification:
            return F.cross_entropy(prediction.flatten(0, 1), target)
        return F.mse_loss(prediction.squeeze(-1).flatten(0, 1), target)

    @torch.inference_mode()
    def _validation_loss(self, tensors: dict[str, torch.Tensor]) -> float:
        self.model.eval()
        prediction = self.model(tensors['x_num'], tensors['x_cat'])
        return float(self._loss(prediction, tensors['y']).detach().cpu())

    @torch.inference_mode()
    def _predict_raw(self, input_data: TensorData) -> np.ndarray:
        X = as_numpy(input_data.features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_num, x_cat = self.preprocessor.transform(X)
        x_num = as_torch(
            x_num, dtype=torch.float32, device=self.device_)
        x_cat = as_torch(x_cat, dtype=torch.long, device=self.device_)

        self.model.eval()
        eval_batch_size = self.params.get('eval_batch_size', 1024)
        predictions = []
        for batch_idx in torch.arange(len(x_num), device=self.device_).split(eval_batch_size):
            predictions.append(self.model(
                x_num[batch_idx], x_cat[batch_idx]).detach().cpu())
        prediction = as_numpy(torch.cat(predictions))

        if self._is_classification:
            probabilities = as_numpy(F.softmax(as_torch(prediction), dim=-1))
            return probabilities.mean(axis=1)
        prediction = prediction.squeeze(-1).mean(axis=1)
        return prediction * self.y_std_ + self.y_mean_

    def _problem_type(self, y) -> str:
        return 'regression'

    def _train_val_split(self, X: np.ndarray, y: np.ndarray):
        val_size = max(
            1, int(round(len(X) * self.params.get('val_fraction', 0.2))))
        val_size = min(val_size, len(X) - 1)
        stratify = None
        if self._is_classification and len(np.unique(y)) > 1:
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2 and val_size >= len(counts):
                stratify = y
        return train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=self.params.get('random_state', 42),
            stratify=stratify,
        )

    def predict(self, input_data: TensorData):
        raw_prediction = self._predict_raw(input_data)
        if self._is_classification:
            prediction = self._decode_prediction(raw_prediction.argmax(axis=1))
        else:
            prediction = raw_prediction
        return self._output(input_data, prediction)

    def predict_proba(self, input_data: TensorData):
        prediction = self._predict_raw(input_data)
        return self._output(input_data, prediction)


class TabMClassifier(TabMImplementation, NeuralTabularClassificationImplementation):
    @property
    def _is_classification(self) -> bool:
        return True

    def _problem_type(self, y) -> str:
        return 'binary' if len(np.unique(y)) == 2 else 'multiclass'


class TabMRegressor(TabMImplementation):
    pass


class RTDLImplementation(NeuralTabularModelImplementation):
    def fit(self, input_data: TensorData):
        self._setup_random_state()
        X = as_numpy(input_data.features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = self._encode_target(self._target(input_data))

        X_train, X_val, y_train, y_val = self._train_val_split(X, y)
        categorical_idx = as_numpy(input_data.categorical_idx)
        if categorical_idx is None:
            categorical_idx = np.array([], dtype=int)
        self.preprocessor = TabMPreprocessor(
            categorical_idx=categorical_idx,
            random_state=self.params.get('random_state', 42),
            num_emb_n_bins=self.params.get(
                'num_emb_n_bins', self._default_num_emb_n_bins()),
        ).fit(X_train)

        x_num_train, x_cat_train = self.preprocessor.transform(X_train)
        x_num_val, x_cat_val = self.preprocessor.transform(X_val)
        train_tensors = self._to_torch_tensors(
            x_num_train, x_cat_train, y_train, fit_regression_target=True)
        val_tensors = self._to_torch_tensors(x_num_val, x_cat_val, y_val)

        self.model = self._make_model(x_num_train.shape[1]).to(self.device_)
        self._fit_model(train_tensors, val_tensors)
        return self

    def _setup_random_state(self):
        seed = self.params.get('random_state', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_num_threads(self.params.get('n_jobs', 1))
        self.device_ = torch.device(self._device_from_params())

    def _device_from_params(self) -> str:
        device = self.params.get('device', 'auto')
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _output_dim(self) -> int:
        if self._is_classification:
            return len(self.classes_)
        return 1

    def _default_num_emb_n_bins(self) -> int:
        return 0

    def _make_model(self, n_num_features: int) -> nn.Module:
        raise NotImplementedError()

    def _forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _to_torch_tensors(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        fit_regression_target: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self._is_classification:
            y_tensor = as_torch(
                y, dtype=torch.long, device=self.device_)
        else:
            y = y.astype(np.float32)
            if fit_regression_target:
                self.y_mean_ = float(np.mean(y))
                self.y_std_ = float(np.std(y))
            y = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            y_tensor = as_torch(
                y, dtype=torch.float32, device=self.device_)

        return {
            'x_num': as_torch(x_num, dtype=torch.float32, device=self.device_),
            'x_cat': as_torch(x_cat, dtype=torch.long, device=self.device_),
            'y': y_tensor,
        }

    def _fit_model(self, train_tensors: dict[str, torch.Tensor], val_tensors: dict[str, torch.Tensor]):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.get('lr', 0.001),
            weight_decay=self.params.get('weight_decay', 0.0001),
        )
        n_train = len(train_tensors['y'])
        batch_size = self._batch_size(n_train)
        best_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        remaining_patience = self.params.get('patience', 6)

        for _ in range(self.params.get('n_epochs', 30)):
            self.model.train()
            for batch_idx in torch.randperm(n_train, device=self.device_).split(batch_size):
                optimizer.zero_grad()
                prediction = self._forward(
                    train_tensors['x_num'][batch_idx], train_tensors['x_cat'][batch_idx])
                loss = self._loss(prediction, train_tensors['y'][batch_idx])
                loss.backward()
                gradient_norm = self.params.get('gradient_clipping_norm', 1.0)
                if gradient_norm is not None and gradient_norm != 'none':
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_norm)
                optimizer.step()

            val_loss = self._validation_loss(val_tensors)
            if val_loss < best_loss:
                best_loss = val_loss
                remaining_patience = self.params.get('patience', 6)
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                remaining_patience -= 1
                if remaining_patience < 0:
                    break

        self.model.load_state_dict(best_state)

    def _batch_size(self, n_train: int) -> int:
        batch_size = self.params.get('batch_size', 'auto')
        if batch_size == 'auto':
            return min(256, n_train)
        return min(int(batch_size), n_train)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._is_classification:
            return F.cross_entropy(prediction, target)
        return F.mse_loss(prediction.squeeze(-1), target)

    @torch.inference_mode()
    def _validation_loss(self, tensors: dict[str, torch.Tensor]) -> float:
        self.model.eval()
        batch_size = self.params.get(
            'eval_batch_size', self._batch_size(len(tensors['y'])))
        total_loss = 0.0
        total_size = 0
        for batch_idx in torch.arange(len(tensors['y']), device=self.device_).split(batch_size):
            prediction = self._forward(
                tensors['x_num'][batch_idx], tensors['x_cat'][batch_idx])
            loss = self._loss(prediction, tensors['y'][batch_idx])
            n_batch = len(batch_idx)
            total_loss += float(loss.detach().cpu()) * n_batch
            total_size += n_batch
        return total_loss / total_size

    @torch.inference_mode()
    def _predict_raw(self, input_data: TensorData) -> np.ndarray:
        X = as_numpy(input_data.features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_num, x_cat = self.preprocessor.transform(X)
        x_num = as_torch(
            x_num, dtype=torch.float32, device=self.device_)
        x_cat = as_torch(x_cat, dtype=torch.long, device=self.device_)

        self.model.eval()
        eval_batch_size = self.params.get('eval_batch_size', 1024)
        predictions = []
        for batch_idx in torch.arange(len(x_num), device=self.device_).split(eval_batch_size):
            predictions.append(self._forward(
                x_num[batch_idx], x_cat[batch_idx]).detach().cpu())
        prediction = as_numpy(torch.cat(predictions))

        if self._is_classification:
            return as_numpy(F.softmax(as_torch(prediction), dim=-1))
        prediction = prediction.squeeze(-1)
        return prediction * self.y_std_ + self.y_mean_

    def _train_val_split(self, X: np.ndarray, y: np.ndarray):
        val_size = max(
            1, int(round(len(X) * self.params.get('val_fraction', 0.2))))
        val_size = min(val_size, len(X) - 1)
        stratify = None
        if self._is_classification and len(np.unique(y)) > 1:
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2 and val_size >= len(counts):
                stratify = y
        return train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=self.params.get('random_state', 42),
            stratify=stratify,
        )

    def predict(self, input_data: TensorData):
        raw_prediction = self._predict_raw(input_data)
        if self._is_classification:
            prediction = self._decode_prediction(raw_prediction.argmax(axis=1))
        else:
            prediction = raw_prediction
        return self._output(input_data, prediction)

    def predict_proba(self, input_data: TensorData):
        prediction = self._predict_raw(input_data)
        return self._output(input_data, prediction)


class FTTransformerImplementation(RTDLImplementation):
    def _make_model(self, n_num_features: int) -> nn.Module:
        from rtdl_revisiting_models import FTTransformer
        return FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=[
                cardinality + 1 for cardinality in self.preprocessor.cat_encoder.cardinalities],
            d_out=self._output_dim(),
            n_blocks=self.params.get('n_blocks', 3),
            d_block=self.params.get('d_block', 64),
            attention_n_heads=self.params.get('attention_n_heads', 4),
            attention_dropout=self.params.get('attention_dropout', 0.1),
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=self.params.get(
                'ffn_d_hidden_multiplier', 4 / 3),
            ffn_dropout=self.params.get('ffn_dropout', 0.1),
            residual_dropout=self.params.get('residual_dropout', 0.0),
        )

    def _forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_num = x_num if x_num.shape[-1] > 0 else None
        x_cat = x_cat if x_cat.shape[-1] > 0 else None
        return self.model(x_num, x_cat)


class FTTransformerClassifier(
    FTTransformerImplementation,
    NeuralTabularClassificationImplementation,
):
    @property
    def _is_classification(self) -> bool:
        return True


class FTTransformerRegressor(FTTransformerImplementation):
    pass


class ResNetImplementation(RTDLImplementation):
    def _make_model(self, n_num_features: int) -> nn.Module:
        from rtdl_revisiting_models import ResNet
        cat_features = sum(
            cardinality + 1 for cardinality in self.preprocessor.cat_encoder.cardinalities)
        d_in = max(n_num_features + cat_features, 1)
        return ResNet(
            d_in=d_in,
            d_out=self._output_dim(),
            n_blocks=self.params.get('n_blocks', 3),
            d_block=self.params.get('d_block', 128),
            d_hidden=None,
            d_hidden_multiplier=self.params.get('d_hidden_multiplier', 2.0),
            dropout1=self.params.get('dropout1', 0.1),
            dropout2=self.params.get('dropout2', 0.1),
        )

    def _forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = []
        if x_num.shape[-1] > 0:
            parts.append(x_num)
        if x_cat.shape[-1] > 0:
            parts.extend([
                F.one_hot(x_cat[:, idx], cardinality + 1).float()
                for idx, cardinality in enumerate(self.preprocessor.cat_encoder.cardinalities)
            ])
        if parts:
            x = torch.cat(parts, dim=1)
        else:
            x = torch.zeros((len(x_num), 1),
                            dtype=torch.float32, device=self.device_)
        return self.model(x)


class ResNetClassifier(ResNetImplementation, NeuralTabularClassificationImplementation):
    @property
    def _is_classification(self) -> bool:
        return True


class ResNetRegressor(ResNetImplementation):
    pass


class RealMLPImplementation(NeuralTabularModelImplementation):
    _excluded_params = {'device', 'n_jobs'}

    def fit(self, input_data: TensorData):
        X = self._to_frame(input_data)
        y = self._encode_target(self._target(input_data))
        model_cls = self._model_cls()

        params = {
            k: v for k, v in self.params.to_dict().items()
            if k not in self._excluded_params
        }
        if params.get('predict_batch_size') == 'auto':
            params['predict_batch_size'] = max(
                min(int(8192 * 200 / max(len(X.columns), 1)), 8192), 64)

        self.model = model_cls(
            device=self._device(input_data),
            n_threads=self.params.get('n_jobs', 1),
            random_state=params.pop('random_state', 42),
            **params,
        )
        self.model.fit(X=X, y=pd.Series(y), cat_col_names=self.cat_col_names)
        return self

    def _model_cls(self):
        from pytabkit import RealMLP_TD_Regressor
        return RealMLP_TD_Regressor

    def predict(self, input_data: TensorData):
        prediction = self._decode_prediction(
            self.model.predict(self._to_frame(input_data)))
        return self._output(input_data, prediction)

    def predict_proba(self, input_data: TensorData):
        prediction = self.model.predict_proba(self._to_frame(input_data))
        return self._output(input_data, prediction)


class RealMLPClassifier(RealMLPImplementation, NeuralTabularClassificationImplementation):
    @property
    def _is_classification(self) -> bool:
        return True

    def _model_cls(self):
        from pytabkit import RealMLP_TD_Classifier
        return RealMLP_TD_Classifier


class RealMLPRegressor(RealMLPImplementation):
    pass
