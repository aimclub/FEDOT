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

sys.setrecursionlimit(1000000000)

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

PERSISTENCE_DIAGRAM_EXTRACTOR = PersistenceDiagramsExtractor(takens_embedding_dim=1,
                                                             takens_embedding_delay=2,
                                                             homology_dimensions=(0, 1),
                                                             parallel=False)


class TopologicalFeaturesImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1

        self.feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PERSISTENCE_DIAGRAM_EXTRACTOR,
            persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

    def fit(self, input_data: InputData):
        pass

    def transform(self, input_data: InputData) -> OutputData:
        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        feature_matrix = parallel(delayed(self.generate_features_from_ts)(sample) for sample in input_data.features)
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

    def generate_features_from_ts(self, ts_data: np.array):
        self.data_transformer = TopologicalTransformation(
            window_length=0)

        point_cloud = self.data_transformer.time_series_to_point_cloud(input_data=ts_data)
        topological_features = self.feature_extractor.transform(point_cloud)
        return topological_features
