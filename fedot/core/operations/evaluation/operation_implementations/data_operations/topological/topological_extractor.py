import math
import sys
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

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
from golem.utilities.utilities import determine_n_jobs

PERSISTENCE_DIAGRAM_FEATURES = {'HolesNumberFeature': HolesNumberFeature(),
                                'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
                                'RelevantHolesNumber': RelevantHolesNumber(),
                                'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
                                'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
                                'PersistenceEntropyFeature': PersistenceEntropyFeature(),
                                'SimultaneousAliveHolesFeature': SimultaneousAliveHolesFeature(),
                                'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
                                'BettiNumbersSumFeature': BettiNumbersSumFeature(),
                                'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()}

PERSISTENCE_DIAGRAM_EXTRACTOR = PersistenceDiagramsExtractor(homology_dimensions=(0, 1))


class TopologicalFeaturesImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_jobs = determine_n_jobs(params.get('n_jobs', 1))
        self.window_size = params.get('window_size')
        # TODO add stride
        self.stride = params.get('stride')
        self.points_count = params.get('points_count')

        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PERSISTENCE_DIAGRAM_EXTRACTOR,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

    def fit(self, input_data: InputData):
        pass

    def transform(self, input_data: InputData) -> OutputData:
        # with Parallel(n_jobs=self.n_jobs) as parallel:
        #     parallel(delayed(self.generate_features_from_ts)(sample) for sample in input_data.features)

        # define window
        if self.window_size == 0:
            # TODO add WindowSizeSelector
            self.window_size = 10

        # define points count
        if self.points_count == 0:
            self.points_count = int(self.window_size // 2)

        time_series = input_data.features
        all_points_cloud = np.array([time_series[i:self.window_size + i]
                                     for i in range(time_series.shape[0] - self.window_size + 1)])

        topological_features = self.feature_extractor.transform(all_points_cloud[:5, :])

        feature_matrix = self.generate_features_from_ts(input_data.features)
        predict = self._clean_predict(np.array([ts for ts in feature_matrix]))
        return predict

    @staticmethod
    def _clean_predict(predict: np.array):
        """Clean predict from nan, inf and reshape data for Fedot appropriate form
        """
        predict = np.where(np.isnan(predict), 0, predict)
        predict = np.where(np.isinf(predict), 0, predict)
        predict = predict.reshape(predict.shape[0], -1)
        return predict
