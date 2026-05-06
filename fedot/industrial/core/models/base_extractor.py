import logging
import math
from multiprocessing import cpu_count

import dask
import torch
from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from numpy.lib import stride_tricks as stride_repr
from pymonad.either import Either
from tqdm.dask import TqdmCallback

from fedot.industrial.core.metrics.metrics_implementation import *
from fedot.industrial.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.repository.constanst_repository import (STAT_METHODS,
                                                                   STAT_METHODS_GLOBAL,
                                                                   STAT_METHODS_TORCH,
                                                                   STAT_METHODS_GLOBAL_TORCH)


class BaseExtractor(IndustrialCachableOperationImplementation):
    """
    Abstract class responsible for feature generator.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.use_cache = self.params.get('use_cache', False)
        self.use_sliding_window = self.params.get('use_sliding_window', True)
        self.use_feature_filter = self.params.get('use_feature_filter', False)
        self.channel_extraction = self.params.get('channel_independent', True)
        self.data_type = DataTypesEnum.table
        self.feature_filter = None
        self.current_window = None
        self.relevant_features = None
        self.predict = None
        self.add_global_features = params.get('add_global_features', True)

        self.stride = 3
        self.n_processes = math.ceil(cpu_count() * 0.7) if cpu_count() > 1 else 1

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging_params = {'jobs': self.n_processes}

        self.is_multichanel = False

    def __repr__(self):
        return 'Abstract Class for TS representation'

    def __check_compute_model(self, input_data: InputData):
        feature_tensor = input_data.features.shape
        is_channel_overrated = feature_tensor[1] > 100
        is_sample_overrated = feature_tensor[0] > 500000
        is_elements_overrated = feature_tensor[2] > 1000
        change_compute_mode = any([is_elements_overrated, is_channel_overrated, is_sample_overrated])
        if change_compute_mode:
            self.channel_extraction = False

    def __check_filter_model(self):
        if self.use_feature_filter and self.feature_filter is not None:
            if not self.feature_filter.is_fitted:
                self.predict = self.feature_filter.reduce_feature_space(self.predict)
            else:
                self.predict = self.predict[:, :, self.feature_filter.feature_mask]

    def fit(self, input_data: InputData):
        pass

    def extract_features(self, x, y) -> pd.DataFrame:
        """
        For those cases when you need to use feature extractor as a standalone object
        """
        transformed_features = self.transform((x, y), use_cache=self.use_cache)
        try:
            return pd.DataFrame(transformed_features.predict.squeeze(), columns=self.relevant_features)
        except ValueError:
            return pd.DataFrame(transformed_features.predict.squeeze())

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        is_tensor_input = isinstance(input_data.features, torch.Tensor)
        if is_tensor_input:
            evaluation_results = self.generate_features_from_ts(input_data.features)
        else:
            evaluation_results = Either(
                value=input_data.features, monoid=[input_data.features, self.channel_extraction]).either(
                left_function=lambda ts_array: self.generate_features_from_array(ts_array),
                right_function=lambda
                ts_array: list(map(lambda sample: self.generate_features_from_ts(sample),
                                   ts_array)))
        if self.channel_extraction and not is_tensor_input:
            with TqdmCallback(desc=fr"compute_feature_extraction_with_{self.__repr__()}"):
                feature_matrix = dask.compute(*evaluation_results)
        else:
            feature_matrix = evaluation_results

        if is_tensor_input:
            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.unsqueeze(1)
            multi_output = feature_matrix.ndim > 2
            self.predict = self._clean_predict_torch(feature_matrix)
            if not multi_output:
                self.predict = self.predict.reshape(self.predict.shape[0], -1)
            self.__check_filter_model()
            return self.predict

        multi_output = len(feature_matrix[0].shape) > 1
        self.predict = Either(value=feature_matrix,
                              monoid=[feature_matrix, multi_output]).either(
            left_function=lambda ts_array: self._clean_predict(np.array(feature_matrix)),
            right_function=lambda matrix: self._clean_predict(np.stack(matrix) if self.channel_extraction
                                                              else np.hstack(feature_matrix)[:, None, :]))
        self.predict = self.predict.reshape(self.predict.shape[0], -1) if not multi_output else self.predict
        self.__check_filter_model()
        return self.predict

    def _clean_predict(self, predict):
        predict = np.where(np.isnan(predict), 0, predict)
        predict = np.where(np.isinf(predict), 0, predict)
        return predict

    def _clean_predict_torch(self, predict: torch.Tensor):
        return torch.nan_to_num(predict, nan=0.0, posinf=0.0, neginf=0.0)

    def generate_features_from_array(self, array: np.array) -> np.array:
        """
        Method responsible for generation of features from time series.
        """
        statistical_representation = self.get_statistical_features(array,
                                                                   add_global_features=self.add_global_features,
                                                                   axis=2)
        return [x for x in statistical_representation if x is not None]

    def get_statistical_features(self, time_series: np.ndarray,
                                 add_global_features: bool = False,
                                 axis=None) -> tuple:
        """
        Method for creating baseline statistical features for a given time series.

        Args:
            add_global_features: if True, global features are added to the feature set
            time_series: time series for which features are generated

        Returns:
            InputData: object with features

        """
        time_series = time_series.flatten() if axis != 2 else time_series
        list_of_methods = [*STAT_METHODS_GLOBAL.items()] if add_global_features else [*STAT_METHODS.items()]
        return list(map(lambda method: method[1](time_series, axis), list_of_methods))

    def get_statistical_features_torch(
            self,
            time_series: torch.Tensor,
            add_global_features: bool = False,
            axis: int = -1,
            max_elements: int = 50000000) -> tuple:
        """
        Method for creating baseline statistical features for a given time series.
        Creates batches, if number of values in time series is more than max_elements.

        Args:
            add_global_features: if True, global features are added to the feature set
            time_series: time series for which features are generated
            max_elements: threshold for using batches

        Returns:
            tuple: features

        """
        if self.is_multichanel:
            if time_series.ndim == 3:
                batch = time_series.shape[0]
                time_series = time_series.reshape(batch, -1)
            elif time_series.ndim > 3:
                batch, b, *rest = time_series.shape
                time_series = time_series.reshape(batch, b, -1)

        if add_global_features:
            list_of_methods = [*STAT_METHODS_GLOBAL_TORCH.items()]
        else:
            list_of_methods = [*STAT_METHODS_TORCH.items()]

        if time_series.numel() <= max_elements:
            return list(map(lambda method: method[1](time_series, axis),
                            list_of_methods))
        B = time_series.shape[0]
        elems_per_sample = time_series[0].numel()
        batch_size = max(1, max_elements // elems_per_sample)
        accumulators = [[] for _ in list_of_methods]
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            ts_batch = time_series[start:end]
            batch_results = [
                method(ts_batch, axis)
                for _, method in list_of_methods
            ]
            for i, res in enumerate(batch_results):
                accumulators[i].append(res)

        merged = []
        for parts in accumulators:
            if isinstance(parts[0], (float, int)):
                merged.append(
                    torch.tensor(parts) if isinstance(parts[0], torch.Tensor) else parts
                )
            else:
                merged.append(torch.cat(parts, dim=0))
        return merged

    def apply_window_for_stat_feature(self,
                                      ts_data: np.array,
                                      feature_generator: callable,
                                      window_size: int = None) -> np.ndarray:
        axis = ts_data.ndim - 1
        if window_size is None:
            window_size = round(ts_data.shape[axis] / 10)
        else:
            window_size = round(ts_data.shape[axis] * (window_size / 100))
        window_size = max(window_size, 5)

        if self.use_sliding_window:
            if self.stride > 1:
                subseq_set = HankelMatrix(time_series=ts_data,
                                          window_size=window_size,
                                          strides=self.stride).trajectory_matrix
            else:
                subseq_set = stride_repr.sliding_window_view(ts_data,
                                                             ts_data.shape[axis] - window_size,
                                                             axis=axis)
        else:
            subseq_set = None

        if subseq_set is None:
            ts_slices = list(range(0, ts_data.shape[0], window_size))
            features = list(map(lambda slice: feature_generator(ts_data[slice:slice + window_size]),
                                ts_slices))
        else:
            ts_slices = list(range(0, subseq_set.shape[1]))
            features = list(map(lambda slice: feature_generator(subseq_set[:, slice]),
                                ts_slices))
        return features

    def _get_feature_matrix(self, extraction_func: callable, ts: np.array) -> np.ndarray:
        multi_channel_features = [extraction_func(x) for x in ts]
        features = np.concatenate([channel_feature.reshape(1, -1)
                                   for channel_feature in multi_channel_features], axis=0)
        return features

    # TODO: refactor with func for ndim > 2
    def _get_torch_feature_matrix(self, extraction_func: callable, ts: torch.Tensor) -> torch.Tensor:
        multi_channel_features = [extraction_func(x) for x in ts]
        features = torch.concat([channel_feature.unsqueeze(0)
                                 for channel_feature in multi_channel_features])
        return features

    def apply_window_for_stat_feature_torch(self,
                                            ts_data: torch.Tensor,
                                            feature_generator: callable,
                                            window_size: int = None) -> torch.Tensor:
        """
        Method for creating windows and extracting base statistical features
        for a given time series or batch of time series.
        """
        axis = ts_data.ndim - 1
        if window_size is None:
            window_size = round(ts_data.shape[axis] / 10)
        else:
            window_size = round(ts_data.shape[axis] * (window_size / 100))
        window_size = max(window_size, 5)

        if self.use_sliding_window:
            if self.stride > 1:
                subseq_set = HankelMatrix(time_series=ts_data,
                                          window_size=window_size,
                                          strides=self.stride).trajectory_matrix
            else:
                window_length = ts_data.shape[axis] - window_size
                subseq_set = ts_data.unfold(
                    dimension=axis,
                    size=window_length,
                    step=self.stride
                )
            if subseq_set.ndim > 2:
                subseq_set = subseq_set.transpose(1, 2)
            else:
                subseq_set = subseq_set.T
        else:
            T = ts_data.shape[1]
            num_windows = T // window_size
            T_eff = num_windows * window_size
            ts_cut = ts_data[:, :T_eff]
            subseq_set = ts_cut.reshape(2, num_windows, window_size)
        features = feature_generator(subseq_set)
        features = torch.stack(features, dim=0).to(ts_data.device)
        if features.ndim > 2:
            features = features.permute(1, 2, 0)
        else:
            features = features.T
        return features
