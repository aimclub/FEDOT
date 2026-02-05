import logging
from copy import deepcopy
from typing import Union

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
from pymonad.either import Either
from sklearn.preprocessing import LabelEncoder

from fedot.industrial.core.architecture.preprocessing.data_convertor import NumpyConverter, DataConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import \
    CURDecomposition
from fedot.industrial.core.operation.dummy.dummy_operation import check_multivariate_data
from fedot.industrial.core.operation.transformation.representation.tabular.tabular_extractor import TabularExtractor
from fedot.industrial.core.repository.config_repository import TASK_MAPPING
from fedot.industrial.core.repository.constanst_repository import FEDOT_DATA_TYPE, fedot_task
from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels


class DataCheck:
    """Class for checking and preprocessing input data for Fedot AutoML.

    Args:
        input_data: Input data in tuple format (X, y) or Fedot InputData object.
        task: Machine learning task, either "classification" or "regression".

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        input_data (InputData): Preprocessed and initialized Fedot InputData object.
        task (str): Machine learning task for the dataset.
        task_dict (dict): Mapping of string task names to Fedot Task objects.

    """

    def __init__(self,
                 input_data: Union[tuple, InputData] = None,
                 task_params: dict = None,
                 task: str = None,
                 fit_stage: bool = False,
                 industrial_task_params=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategy_params = industrial_task_params
        self.convert_ts_method = {'ts2tabular': self._convert_ts2tabular,
                                  'ts2image': self._convert_ts2image,
                                  'big_dataset': self._convert_big_data}
        if hasattr(industrial_task_params, 'strategy_params'):
            self.strategy_params = industrial_task_params.strategy_params
            self.manager = industrial_task_params
        self.data_type = FEDOT_DATA_TYPE[self.strategy_params['data_type']] \
            if self.strategy_params is not None else FEDOT_DATA_TYPE['tensor']

        self.input_data = input_data
        self.data_convertor = DataConverter(data=self.input_data)
        self.is_already_fedot_type = isinstance(self.input_data, InputData)
        self.task = task
        self.task_params = task_params if task_params is not None else {}
        self.fit_stage = fit_stage
        self.label_encoder = None

    def __check_features_and_target(self, input_data, data_type):
        if data_type == 'torchvision':
            X, multi_features, y = input_data[0].data.cpu().detach(
            ).numpy(), True, input_data[0].targets.cpu().detach().numpy()
        elif self.data_convertor.is_tuple:
            X, y = input_data[0], input_data[1]
        else:
            X, y = input_data.features, input_data.target

        multi_features, features = check_multivariate_data(X)
        multi_target = len(y.shape) > 1 and y.shape[1] > 2
        target = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        target = target.reshape(-1, 1) if multi_features and not multi_target else np.ravel(target).reshape(-1, 1)
        data_dict = dict(features=features,
                         target=target,
                         multi_features=multi_features,
                         multi_target=multi_target)
        return data_dict

    def _encode_target(self, data_dict):
        self.label_encoder = LabelEncoder()
        data_dict['target'] = self.label_encoder.fit_transform(data_dict['target'])
        return data_dict

    def _transformation_for_ts_forecasting(self, data_dict):
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(
            forecast_length=self.task_params['forecast_length']))
        if self.data_convertor.is_numpy_matrix and any(
                [self.data_convertor.have_one_sample, self.data_convertor.have_one_channel]):
            features_array = self.data_convertor.convert_to_1d_array()
        else:
            features_array = self.data_convertor.numpy_data

        if self.fit_stage:
            features_array = features_array[:-self.task_params['forecast_length']]
            target = features_array[-self.task_params['forecast_length']:]
        else:
            features_array = features_array
            target = features_array
        return InputData.from_numpy_time_series(
            features_array=features_array, target_array=target, task=task)

    def _transformation_for_other_task(self, data_dict: dict = None):
        def encode_target(init_dict): return Either(
            value=init_dict,
            monoid=[
                init_dict,
                self.label_encoder is not None]).either(
            left_function=lambda dict: dict,
            right_function=lambda dict: self._encode_target(dict))

        def encode_idx(dict_with_target): return Either(value=dict_with_target,
                                                        monoid=[dict_with_target,
                                                                not self.data_convertor.is_torchvision_dataset]).either(
            right_function=lambda dict: dict | {'idx': np.arange(dict['features'].shape[0])},
            left_function=lambda dict: dict | {'idx': np.arange(len(dict['features'][0]))})

        def define_horizon(dict_with_idx): return Either(value=dict_with_idx,
                                                         monoid=[dict_with_idx, self.strategy_params is None]).either(
            left_function=lambda dict: dict | {'have_predict_horizon':
                                               all([self.strategy_params['data_type'] == 'time_series',
                                                    'detection_window' in self.strategy_params.keys()])},
            right_function=lambda dict: dict | {'have_predict_horizon': False})

        def define_task(dict_with_horizon): return Either(value=dict_with_horizon,
                                                          monoid=[dict_with_horizon,
                                                                  dict_with_horizon['have_predict_horizon']]).either(
            right_function=lambda dict: dict |
            {'task': fedot_task('ts_forecasting',
                                self.strategy_params['detection_window'])},
            left_function=lambda dict: dict | {'task': fedot_task(self.task)})

        encoded_dict = Either.insert(data_dict). \
            then(lambda data: encode_target(data)). \
            then(lambda data_with_target: encode_idx(data_with_target)). \
            then(lambda data_with_idx: define_horizon(data_with_idx)). \
            then(lambda data_with_horizon: define_task(data_with_horizon)).value

        return InputData(idx=encoded_dict['idx'],
                         features=encoded_dict['features'],
                         target=encoded_dict['target'],
                         task=encoded_dict['task'],
                         data_type=self.data_type)

    def _init_input_data(self) -> None:
        """Initializes the `input_data` attribute based on its type.

        If a tuple (X, y) is provided, it converts it to a Fedot InputData object
        with appropriate data types and task information. If an existing InputData
        object is provided, it checks if it requires further initialization.

        Raises:
            ValueError: If the input data format is invalid.

        """

        def non_forecasting_transformation(data): return \
            Either(value=data, monoid=[data, self.data_convertor.is_tuple]).either(
                right_function=lambda r: self.__check_features_and_target(r, 'tuple'),
                left_function=lambda l: self.__check_features_and_target(l, 'torchvision'))

        self.input_data = Either(value=dict(features=self.input_data,
                                            multivariate=False,
                                            target=self.input_data),
                                 monoid=[non_forecasting_transformation,
                                         self.task == 'ts_forecasting']).either(
            right_function=lambda data_dict: self._transformation_for_ts_forecasting(data_dict),
            left_function=lambda transformation_func: self._transformation_for_other_task(
                transformation_func(self.input_data)))

    def _check_input_data_features(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        self.input_data.features = Either.insert(self.input_data.features). \
            then(lambda data: np.where(np.isnan(data), 0, data)). \
            then(lambda data_without_nan: np.where(np.isinf(data_without_nan), 0, data_without_nan)). \
            then(lambda data_without_inf: NumpyConverter(data=data_without_inf).convert_to_torch_format()
                 if self.task != 'ts_forecasting' else data_without_inf).value

    def _check_input_data_target(self):
        """Checks and preprocesses the features in the input data.

        - Replaces NaN and infinite values with 0.
        - Converts features to torch format using NumpyConverter.

        """
        if self.task == 'regression':
            self.input_data.target = self.input_data.target.squeeze().astype(float)
        elif self.task == 'classification':
            self.input_data.target[self.input_data.target == -1] = 0
        else:
            _ = 1

    def _check_fedot_context(self):
        if self.strategy_params is not None:
            IndustrialModels().setup_repository()
            strategy = self.strategy_params.get('learning_strategy')
            is_big_data = strategy.__contains__('big') if strategy else False
            is_default_fedot = strategy.__contains__('tabular') if strategy else False
            output_data = Either(value=self.strategy_params.get('learning_strategy', None),
                                 monoid=[self.input_data, is_default_fedot]).either(
                left_function=lambda x: x.features,
                right_function=lambda strategy: self.convert_ts_method[strategy]
                (self.input_data, self.strategy_params.get('sampling_strategy', None)))
            self.input_data.features = output_data.predict if hasattr(output_data, 'predict') else output_data
            if is_big_data and hasattr(output_data, 'target'):
                self.input_data.target = output_data.target

    def _convert_ts2tabular(self, input_data, sampling_strategy):
        if sampling_strategy is not None:
            sample_start, sample_end = list(sampling_strategy['samples'].values())
            channel_start, channel_end = list(sampling_strategy['channels'].values())
            element_start, element_end = list(sampling_strategy['elements'].values())
            input_data.features = self.input_data.features[
                sample_start:sample_end,
                channel_start:channel_end,
                element_start:element_end]
        fg_list = self.strategy_params['feature_generator']
        ts2tabular_model = TabularExtractor({'feature_domain': fg_list,
                                             'reduce_dimension': False})
        return ts2tabular_model.transform(input_data)

    def _convert_ts2image(self):
        pass

    def _convert_big_data(self, input_data, sampling_strategy: dict):
        approx_method_dict = {'CUR': CURDecomposition}
        approx_method, method_params = list(sampling_strategy.items())[0]
        big_dataset_model = approx_method_dict[approx_method](method_params)
        return big_dataset_model.transform(input_data)

    def check_available_operations(self, available_operations):
        pass

    def _process_input_data(self):
        self._init_input_data()
        if not self.data_convertor.is_torchvision_dataset:
            self._check_input_data_features()
            self._check_input_data_target()
            self._check_fedot_context()
        self.input_data.supplementary_data.is_auto_preprocessed = True

        return self.input_data

    def check_input_data(self) -> InputData:
        """Checks and preprocesses the input data for Fedot AutoML.

        Performs the following steps:
            1. Initializes the `input_data` attribute based on its type.
            2. Checks and preprocesses the features (replacing NaNs, converting to torch format).
            3. Checks and preprocesses the target variable (encoding labels, casting to float).

        Returns:
            InputData: The preprocessed and initialized Fedot InputData object.

        """

        return self.input_data if self.is_already_fedot_type else self._process_input_data()

    def get_target_encoder(self):
        return self.label_encoder


class ApiConfigCheck:
    def __init__(self):
        pass

    def compare_configs(self, original, updated):
        """Compares two nested dictionaries"""

        changes = []

        def recursive_compare(orig, upd, path):
            all_keys = orig.keys() | upd.keys()
            for key in all_keys:
                orig_val = orig.get(key, "<MISSING>")
                upd_val = upd.get(key, "<MISSING>")

                if isinstance(orig_val, dict) and isinstance(upd_val, dict):
                    recursive_compare(orig_val, upd_val, path + [key])
                elif orig_val != upd_val:
                    changes.append(f"{' -> '.join(map(str, path + [key]))} -> Changed value {orig_val} to {upd_val}")

        for sub_config in original.keys():
            if sub_config in updated:
                recursive_compare(original[sub_config], updated[sub_config], [sub_config])
            else:
                changes.append(f"{sub_config} -> Removed completely")

        for i in changes:
            print('>>>', i)
        return "\n".join(changes) if changes else "No changes detected."

    def update_config_with_kwargs(self, config_to_update, **kwargs):
        """ Recursively update config dictionary with provided keyword arguments. """

        # prevent inplace changes to the original config
        config = deepcopy(config_to_update)

        def recursive_update(d, key, value):
            if key in d:
                d[key] = value
                # print(f'Updated {key} with {value}')
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_update(v, key, value)

        # we select automl problem
        assert 'task' in kwargs, 'Problem type is not provided'
        problem_type = kwargs['task']
        config['automl_config'] = TASK_MAPPING[problem_type]

        # change MEGA config with keyword arguments
        for param, value in kwargs.items():
            recursive_update(config, param, value)

        return config
