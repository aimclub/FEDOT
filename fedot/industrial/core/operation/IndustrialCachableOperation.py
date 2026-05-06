import os
from typing import Optional

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.caching import DataCacher
from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH


class IndustrialCachableOperationImplementation(DataOperationImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        cache_folder = os.path.join(PROJECT_PATH, 'cache')
        os.makedirs(cache_folder, exist_ok=True)
        self.cacher = DataCacher(data_type_prefix='Features of basis',
                                 cache_folder=cache_folder)
        self.channel_extraction = True
        self.data_type = DataTypesEnum.image
        self.hashable_attr = ['stride',
                              'window_size',
                              'add_global_features',
                              'use_sliding_window',
                              'channel_independent']

    def __check_compute_model(self, input_data: InputData):
        feature_tensor = input_data.features.shape
        if len(feature_tensor) > 2:
            is_channel_overrated = feature_tensor[1] > 100
            is_sample_overrated = feature_tensor[0] > 500000
            is_elements_overrated = feature_tensor[2] > 1000
            if any([is_elements_overrated, is_channel_overrated, is_sample_overrated]):
                self.channel_extraction = False

    def _create_hash_descr(self):
        return {k: v for k, v in self.dict_keys.items() if k in self.hashable_attr}

    def _convert_to_fedot_datatype(
            self,
            input_data=None,
            transformed_features=None):
        if not isinstance(input_data, InputData):
            input_data = InputData(idx=np.arange(len(transformed_features)),
                                   features=transformed_features,
                                   target='no_target',
                                   task='no_task',
                                   data_type=DataTypesEnum.table)

        if isinstance(transformed_features, OutputData):
            transformed_features = transformed_features.predict

        predict = OutputData(idx=input_data.idx,
                             features=input_data.features,
                             features_names=input_data.features_names,
                             predict=transformed_features,
                             task=input_data.task,
                             target=input_data.target,
                             data_type=input_data.data_type,
                             supplementary_data=input_data.supplementary_data)
        return predict

    def fit(self, data):
        """Decomposes the given data on the chosen basis.

        Returns:
            np.array: The decomposition of the given data.
        """

    def try_load_from_cache(self, hashed_info: str) -> np.array:
        predict = self.cacher.load_data_from_cache(hashed_info=hashed_info)
        return predict

    def transform(
            self,
            input_data: InputData,
            use_cache: bool = True) -> OutputData:
        """Method firstly tries to load result from cache. If unsuccessful, it starts to generate features

        Args:
            input_data: InputData - data to transform
            use_cache: bool - whether to use cache or not

        Returns:
            OutputData - transformed data

        """
        self.__check_compute_model(input_data)
        if use_cache:
            self.dict_keys = {k: v for k, v in self.__dict__.items()}
            class_params = self._create_hash_descr()
            class_params['model'] = self.__repr__()
            class_params['input_data_shape'] = input_data.features.shape
            hashed_info = self.cacher.hash_info(operation_info=class_params.__repr__())
            try:
                transformed_features = self.try_load_from_cache(hashed_info)
            except (FileNotFoundError, ValueError):
                transformed_features = self._transform(input_data)
                self.cacher.cache_data(hashed_info, transformed_features)

            predict = self._convert_to_fedot_datatype(input_data, transformed_features)
            return predict
        else:
            transformed_features = self._transform(input_data)
            predict = self._convert_to_fedot_datatype(input_data, transformed_features)
            return predict

    def _transform(self, input_data):
        pass
