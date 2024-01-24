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
        self.points_count = params.get('points_count')
        self.max_homology_dimension = params.get('max_homology_dimension')
        self.metric = params.get('metric')
        self.n_jobs = params.get('n_jobs')
        self.feature_funs = (lambda x: np.quantile(x, (0.1, 0.25, 0.5, 0.75, 0.9)), )
        self._shape = None

    def fit(self, input_data: InputData):
        if self.points_count == 0:
            self.points_count = int(input_data.features.shape[1] * 0.33)

        # define shape of features after transforming on the one data sample
        sample = input_data.features[0, :].ravel()
        features = np.concatenate([fun(sample) for fun in self.feature_funs])
        self._shape = features.shape[0]
        return self

    def transform(self, input_data: InputData) -> OutputData:
        with Parallel(n_jobs=self.n_jobs, prefer='processes') as parallel:
            topological_features = parallel(delayed(self._extract_features)(data)
                                            for data in input_data.features)
        return np.array(topological_features)

    def _extract_features(self, x):
        x_sliced = [x[i:self.points_count + i] for i in range(x.shape[0] - self.points_count + 1)]
        x_processed = ripser(x_sliced,
                             maxdim=self.max_homology_dimension,
                             coeff=2,
                             metric=self.metric,
                             n_threads=1,
                             collapse_edges=False)["dgms"]
        result = list()
        for xp in x_processed:
            if xp.shape[0] > 0:
                xp = xp[:, 1] - xp[:, 0]
                for fun in self.feature_funs:
                    result.append(fun(xp))
            else:
                result.append(np.zeros(self._shape))
        return np.concatenate(result)
