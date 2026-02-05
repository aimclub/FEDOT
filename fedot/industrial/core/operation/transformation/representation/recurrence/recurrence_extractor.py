from typing import Optional

import dask
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

# from fedot.industrial.core.metrics.metrics_implementation import *
from fedot.industrial.core.models.base_extractor import BaseExtractor
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.data.kernel_matrix import TSTransformer
from fedot.industrial.core.operation.transformation.representation.recurrence.sequences import RecurrenceFeatureExtractor


class RecurrenceExtractor(BaseExtractor):
    """Class responsible for wavelet feature generator experiment.

    Attributes:
        transformer: TSTransformer object.
        self.extractor: RecurrenceExtractor object.
        self.window_mode: bool, if True, then the window mode is used.
        self.min_signal_ratio: float, the minimum signal ratio.
        self.max_signal_ratio: float, the maximum signal ratio.
        self.rec_metric: str, the metric for calculating the recurrence matrix.
        self.window_size: int, the window size.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('eigen_basis').add_node('recurrence_extractor').add_node(
                    'rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        # TODO add threshold for other metrics
        self.rec_metric = params.get('rec_metric', 'cosine')
        self.image_mode = params.get('image_mode', False)
        self.transformer = TSTransformer
        self.extractor = RecurrenceFeatureExtractor

    def __repr__(self):
        return 'Reccurence Class for TS representation'

    def _generate_features_from_ts(self, ts: np.array):
        if self.window_size != 0:
            trajectory_transformer = HankelMatrix(time_series=ts,
                                                  window_size=self.window_size,
                                                  strides=self.stride)
            ts = trajectory_transformer.trajectory_matrix
            self.ts_length = trajectory_transformer.ts_length

        specter = self.transformer(time_series=ts,
                                   rec_metric=self.rec_metric)

        if not self.image_mode:
            feature_df = specter.ts_to_recurrence_matrix()
            feature_df = self.extractor(
                recurrence_matrix=feature_df).quantification_analysis()
            features = np.nan_to_num(
                np.array(list(feature_df.values())), posinf=0, neginf=0)
        else:
            features = specter.ts_to_3d_recurrence_matrix()

        return features

    def generate_recurrence_features(self, ts: np.array) -> InputData:

        if len(ts.shape) < 3:
            aggregation_df = self._generate_features_from_ts(ts)
        else:
            aggregation_df = self._get_feature_matrix(
                self._generate_features_from_ts, ts)

        return aggregation_df

    @dask.delayed
    def generate_features_from_ts(self, ts_data: np.array,
                                  dataset_name: str = None):
        return self.generate_recurrence_features(ts=ts_data)
