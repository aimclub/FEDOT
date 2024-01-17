import math
import sys
from itertools import chain
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
from gph import ripser_parallel as ripser

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.point_cloud import \
    TopologicalTransformation
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.topological import \
    HolesNumberFeature, MaxHoleLifeTimeFeature, RelevantHolesNumber, AverageHoleLifetimeFeature, \
    SumHoleLifetimeFeature, PersistenceEntropyFeature, SimultaneousAliveHolesFeature, \
    AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, RadiusAtMaxBNFeature, PersistenceDiagramsExtractor, \
    TopologicalFeaturesExtractor
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.window_size_selector import WindowSizeSelector, WindowSizeSelectorMethodsEnum
from golem.utilities.utilities import determine_n_jobs


class FastTopologicalFeaturesImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size')
        self.points_count = params.get('points_count')
        self.max_homology_dimension = 1
        self.shapes = None

    def fit(self, input_data: InputData):
        if self.points_count == 0:
            self.points_count = int(input_data.features.shape[1] * 0.33)

        self.shapes = None

        return self

    def transform(self, input_data: InputData) -> OutputData:
        topological_features = [self._extract_features(self._slice_by_window(data, self.points_count))
                                for data in input_data.features]

        if self.shapes is None:
            self.shapes = [max(x[dim].shape[0] for x in topological_features)
                           for dim in range(self.max_homology_dimension + 1)]

        features = list()
        for dim in range(self.max_homology_dimension + 1):
            _features = np.zeros((len(topological_features), self.shapes[dim]))
            for topo_features_num, topo_features in enumerate(topological_features):
                if len(topo_features[dim]) > 0:
                    x = topo_features[dim][:, 1] - topo_features[dim][:, 0]
                    _features[topo_features_num, :len(x)] = x
            features.append(_features)
        features = np.concatenate(features, axis=1)
        features[np.isinf(features) | (features < 0)] = 0

        return features

    def _extract_features(self, x):
        x_processed = ripser(x,
                             maxdim=self.max_homology_dimension,
                             coeff=2,
                             metric='euclidean',
                             n_threads=1,
                             collapse_edges=False)["dgms"]
        return x_processed

    def _slice_by_window(self, data, window):
        return [data[i:window + i] for i in range(data.shape[0] - window + 1)]