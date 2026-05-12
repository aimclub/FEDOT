from typing import Optional

import dask
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.base_extractor import BaseExtractor


class QuantileExtractor(BaseExtractor):
    """Class responsible for statistical feature generator experiment.

    Attributes:
        window_size (int): size of window
        stride (int): stride for window
        var_threshold (float): threshold for variance

    Example:
        To use this class you need to import it and call needed methods::

            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('quantile_extractor',
                                                       params={'window_size': 20, 'window_mode': True})
                                            .add_node('rf')
                                            .build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = params.get('window_size', 0)
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.logging_params.update({'Wsize': self.window_size,
                                    'Stride': self.stride})

    def _concatenate_global_and_local_feature(
            self,
            global_features: np.ndarray,
            window_stat_features: np.ndarray) -> np.ndarray:
        if isinstance(window_stat_features[0], list):
            window_stat_features = np.concatenate(window_stat_features, axis=0)

        window_stat_features = np.concatenate(
            [global_features, window_stat_features], axis=0)
        window_stat_features = np.nan_to_num(window_stat_features)
        return window_stat_features

    def extract_stats_features(self, ts: np.array, axis: int) -> InputData:
        global_features = self.get_statistical_features(
            ts, add_global_features=self.add_global_features, axis=axis)
        window_stat_features = self.get_statistical_features(ts, axis=axis) if self.window_size == 0 else \
            self.apply_window_for_stat_feature(ts_data=ts, feature_generator=self.get_statistical_features,
                                               window_size=self.window_size)
        return self._concatenate_global_and_local_feature(
            global_features, window_stat_features) if self.add_global_features else window_stat_features

    @dask.delayed
    def generate_features_from_ts(self,
                                  ts: np.array,
                                  window_length: int = None) -> InputData:
        # sanity check for map method
        ts = ts[None, :] if len(ts.shape) == 1 else ts
        statistical_representation = np.array(
            list(map(lambda channel: self.extract_stats_features(channel, axis=0), ts)))
        return statistical_representation

    def generate_features_from_array(self, array: np.array) -> InputData:
        statistical_representation = self.get_statistical_features(array,
                                                                   add_global_features=self.add_global_features, axis=2)
        return [x for x in statistical_representation if x is not None]
