import os
from typing import Optional

import numpy as np
import torch

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir


class BaseFoundationalImplementation(ModelImplementation):
    _excluded_model_params = [
        'max_samples',
        'max_features',
        'device'
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model_params = {
            k: v for k, v in self.params.to_dict().items() if k not in self._excluded_model_params
        }
        self.model = None
        self.classes_ = None

    def _prepare_features(self, data: TensorData, *, fit: bool = False):
        # Public fit/predict APIs use CPU NumPy/sklearn preprocessing even on GPU.
        return as_numpy(data.features)

    def fit(self, input_data: TensorData):
        features = self._prepare_features(input_data, fit=True)
        target = input_data.target
        if target.ndim == 2 and target.shape[1] == 1:
            target = target.reshape(-1)
        self.model.fit(X=features, y=as_numpy(target))
        return self.model

    def predict(self, input_data: TensorData) -> torch.Tensor:
        prediction = self.model.predict(self._prepare_features(input_data))
        return as_torch(prediction, device=input_data.device)

    def predict_proba(self, input_data: TensorData) -> torch.Tensor:
        prediction = self.model.predict_proba(self._prepare_features(input_data))
        return as_torch(prediction, device=input_data.device)


class BaseFoundationalClassificationImplementation(BaseFoundationalImplementation):
    def fit(self, input_data: TensorData):
        self.classes_ = torch.unique(input_data.target)
        return super().fit(input_data=input_data)


class FedotTabPFNImplementation(BaseFoundationalImplementation):
    """FEDOT implementation of TabPFN that uses tabpfn model version 2.
    To use the newer TabPFN v2.5 model, you must accept the non-commercial usage license
    in your Hugging Face account at https://huggingface.co/Prior-Labs/tabpfn_2_5.
    After accepting the license, update the `tabpfn` library to the latest version
    (currently 6.0.6) and log in to your Hugging Face account, for example using env variable:
    `os.environ["HF_TOKEN"] = "your_hf_token"`. The library will then automatically
    use the v2.5 model. If authentication with a token fails, you will get an error.

    """

    _excluded_model_params = [
        'enable_categorical',
        'max_samples',
        'max_features',
        'model_path'
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        model_path = self.params.get('model_path', None)
        if model_path == "auto":
            model_path = os.path.join(default_fedot_data_dir(), 'tabpfn')
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path
        elif model_path is not None:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path

    def _prepare_features(self, data: TensorData, *, fit: bool = False):
        if not self.params.get('enable_categorical'):
            if fit:
                self.model.categorical_features_indices = data.categorical_idx
            return super()._prepare_features(data, fit=fit)

        original = data.get_not_encoded_data()
        numerical = as_numpy(original.numerical_features)
        categories = original.categorical_features
        n_numerical, n_categorical = numerical.shape[1], categories.shape[1]
        if fit:
            # Restoration puts categories after numerical columns, unlike OHE layout.
            self.model.categorical_features_indices = list(range(n_numerical, n_numerical + n_categorical))
        if n_categorical == 0:
            return numerical
        # Host strings require a mixed array at the library boundary. Keep missing
        # values intact: optional TensorData imputation has already decided their policy.
        features = np.empty((len(data.features), n_numerical + n_categorical), dtype=object)
        features[:, :n_numerical] = numerical
        features[:, n_numerical:] = categories
        return features


class FedotTabPFNClassificationImplementation(
    FedotTabPFNImplementation,
    BaseFoundationalClassificationImplementation
):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(**self.model_params)


class FedotTabPFNRegressionImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabpfn import TabPFNRegressor
        self.model = TabPFNRegressor(**self.model_params)


class FedotTabICLImplementation(BaseFoundationalImplementation):
    _excluded_model_params = [
        'max_samples',
        'max_features'
    ]


class FedotTabICLClassificationImplementation(
    FedotTabICLImplementation,
    BaseFoundationalClassificationImplementation
):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabicl import TabICLClassifier
        self.model = TabICLClassifier(**self.model_params)


class FedotTabICLRegressionImplementation(FedotTabICLImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabicl import TabICLRegressor
        self.model = TabICLRegressor(**self.model_params)
