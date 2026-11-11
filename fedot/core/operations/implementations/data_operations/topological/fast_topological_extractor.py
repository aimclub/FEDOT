from typing import Optional

import numpy as np
import torch

from joblib import Parallel, delayed

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.operations.implementations.data_operations.tools import replace_projected_features
from fedot.core.operations.implementations.data_operations.topological.topological_backend import (
    ripser,
    TOPOLOGICAL_BACKEND_AVAILABLE,
)
from fedot.core.operations.operation_parameters import OperationParameters


class TopologicalFeaturesImplementation(TensorDataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size_as_share = params.get('window_size_as_share')
        self.max_homology_dimension = params.get('max_homology_dimension')
        self.metric = params.get('metric')
        self.stride = params.get('stride')
        self.n_jobs = params.get('n_jobs')
        self.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
        self._shape = len(self.quantiles)
        self._window_size = None

    def fit(self, input_data: TensorData):
        self._window_size = int(
            input_data.features.shape[1] * self.window_size_as_share)
        self._window_size = max(self._window_size, 2)
        self._window_size = min(
            self._window_size, input_data.features.shape[1] - 2)
        self._window_size = max(self._window_size, 1)
        return self

    def transform(self, input_data: TensorData) -> TensorData:
        if not TOPOLOGICAL_BACKEND_AVAILABLE:
            raise ModuleNotFoundError("Install topological dependencies with 'pip install fedot[extra]'")
        features = input_data.features
        signals = torch.stack([features[i:i + 2, ::self.stride].mean(dim=0)
                               for i in range(0, features.shape[0], 2)])
        windows = as_numpy(signals.unfold(1, self._window_size, 1))
        with Parallel(n_jobs=self.n_jobs, prefer='processes') as parallel:
            topological_features = parallel(delayed(self._extract_features)(window) for window in windows)
        result = as_torch(np.stack(topological_features), device=input_data.device)
        result = result.repeat_interleave(2, dim=0)[:features.shape[0]]
        result = torch.nan_to_num(result, nan=0, posinf=0, neginf=0)
        return replace_projected_features(input_data, result)

    def _extract_features(self, x_sliced):
        x_processed = ripser(x_sliced,
                             maxdim=self.max_homology_dimension,
                             coeff=2,
                             metric=self.metric)["dgms"]
        result = np.zeros(self._shape * (self.max_homology_dimension + 1))
        for i, xp in enumerate(x_processed):
            if xp.shape[0] > 0:
                result[i * self._shape:(i + 1) * self._shape] = np.quantile(xp[:, 1] - xp[:, 0], self.quantiles,
                                                                            overwrite_input=True, method='hazen')
        return result
