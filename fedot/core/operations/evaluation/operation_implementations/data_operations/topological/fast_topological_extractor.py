from typing import Optional

import numpy as np
from gph import ripser_parallel as ripser

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FastTopologicalFeaturesImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.points_count = params.get('points_count')
        self.max_homology_dimension = 1
        self.feature_funs = (lambda x: np.quantile(x, (0.1, 0.25, 0.5, 0.75, 0.9)), )
        self.shape = None

    def fit(self, input_data: InputData):
        if self.points_count == 0:
            self.points_count = int(input_data.features.shape[1] * 0.33)
        self.shape = sum(map(len, [fun(np.zeros((10, ))) for fun in self.feature_funs]))
        return self

    def transform(self, input_data: InputData) -> OutputData:
        topological_features = [self._extract_features(self._slice_by_window(data, self.points_count))
                                for data in input_data.features]
        return np.array(topological_features)

    def _extract_features(self, x):
        x_processed = ripser(x,
                             maxdim=self.max_homology_dimension,
                             coeff=2,
                             metric='euclidean',
                             n_threads=1,
                             collapse_edges=False)["dgms"]
        result = list()
        for xp in x_processed:
            if xp.shape[0] > 0:
                xp = xp[:, 1] - xp[:, 0]
                for fun in self.feature_funs:
                    result.append(fun(xp))
            else:
                result.append(np.zeros(self.shape))
        return np.concatenate(result)

    def _slice_by_window(self, data, window):
        return [data[i:window + i] for i in range(data.shape[0] - window + 1)]
