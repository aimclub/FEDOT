import logging
from itertools import chain
from typing import Optional

import numpy as np

try:
    from gph import ripser_parallel as ripser
except ModuleNotFoundError:
    logging.log(100,
                "Topological features operation requires extra dependencies for time series forecasting, which are not installed. It can infuence the performance. Please install it by 'pip install fedot[extra]'")

from joblib import Parallel, delayed

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class TopologicalFeaturesImplementation(DataOperationImplementation):
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

    def fit(self, input_data: InputData):
        self._window_size = int(input_data.features.shape[1] * self.window_size_as_share)
        self._window_size = max(self._window_size, 2)
        self._window_size = min(self._window_size, input_data.features.shape[1] - 2)
        return self

    def transform(self, input_data: InputData) -> OutputData:
        features = input_data.features
        with Parallel(n_jobs=self.n_jobs, prefer='processes') as parallel:
            topological_features = parallel(delayed(self._extract_features)
                                            (np.mean(features[i:i + 2, ::self.stride], axis=0))
                                            for i in range(0, features.shape[0], 2))
        if len(topological_features) * 2 < features.shape[0]:
            topological_features.append(topological_features[-1])
        result = np.array(list(chain(*zip(topological_features, topological_features))))
        if result.shape[0] > features.shape[0]:
            result = result[:-1, :]
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        return result

    def _extract_features(self, x):
        x_sliced = np.array([x[i:self._window_size + i] for i in range(x.shape[0] - self._window_size + 1)])
        x_processed = ripser(x_sliced,
                             maxdim=self.max_homology_dimension,
                             coeff=2,
                             metric=self.metric,
                             n_threads=1,
                             collapse_edges=False)["dgms"]
        result = np.zeros(self._shape * (self.max_homology_dimension + 1))
        for i, xp in enumerate(x_processed):
            if xp.shape[0] > 0:
                result[i * self._shape:(i + 1) * self._shape] = np.quantile(xp[:, 1] - xp[:, 0], self.quantiles,
                                                                            overwrite_input=True, method='hazen')
        return result
