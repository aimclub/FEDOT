from typing import Optional

import numpy as np
from gph import ripser_parallel as ripser
from joblib import Parallel, delayed

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FastTopologicalFeaturesImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size_as_share = params.get('window_size_as_share')
        self.max_homology_dimension = params.get('max_homology_dimension')
        self.metric = params.get('metric')
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
        with Parallel(n_jobs=self.n_jobs, prefer='processes') as parallel:
            topological_features = parallel(delayed(self._extract_features)(data)
                                            for data in input_data.features)
        result = np.array(topological_features)
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        return result

    def _extract_features(self, x):
        x_sliced = [x[i:self._window_size + i] for i in range(x.shape[0] - self._window_size + 1)]
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
