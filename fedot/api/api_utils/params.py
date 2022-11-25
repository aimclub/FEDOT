import random
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.constants import AUTO_PRESET_NAME, DEFAULT_FORECAST_LENGTH
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum, TsForecastingParams
from fedot.core.utilities.random import RandomStateHandler


class ApiParams:

    def __init__(self):
        self.api_params = None
        self.log = None
        self.task = None
        self.metric_to_compose = None
        self.metric_to_tuning = None
        self.task_params = None
        self.metrics_name = None

    def initialize_params(self, input_params: Dict[str, Any]):
        """ Merge input_params dictionary with several parameters for AutoML algorithm """
        self.get_initial_params(input_params)

        # Final check for correctness for timeout and generations
        self.api_params = check_timeout_vs_generations(self.api_params)

    def update_available_operations_by_preset(self, data: InputData):
        preset_operations = OperationsPreset(task=self.task, preset_name=self.api_params['preset'])
        self.api_params = preset_operations.composer_params_based_on_preset(self.api_params, data.data_type)

    def get_initial_params(self, input_params: Dict[str, Any]):
        self._parse_input_params(input_params)

        param_dict = {
            'task': self.task,
            'logger': self.log
        }
        self.api_params = {**self.api_params, **param_dict}

    def accept_and_apply_recommendations(self, input_data: Union[InputData, MultiModalData], recommendations: Dict):
        """
        Accepts recommendations for api params from DataAnalyser

        :param input_data - data for preprocessing
        :param recommendations - dict with recommendations
        """
        # TODO fix multimodality
        if isinstance(input_data, MultiModalData):
            self.api_params['cv_folds'] = None  # there are no support for multimodal data now
            for data_source_name, values in input_data.items():
                self.accept_and_apply_recommendations(input_data[data_source_name],
                                                      recommendations[data_source_name])
        else:
            if 'label_encoded' in recommendations:
                self.log.info("Change preset due to label encoding")
                self.change_preset_for_label_encoded_data(input_data.task, input_data.data_type)

    def change_preset_for_label_encoded_data(self, task: Task, data_type: DataTypesEnum):
        """ Change preset on tree like preset, if data had been label encoded """
        if 'preset' in self.api_params:
            preset_name = ''.join((self.api_params['preset'], '*tree'))
        else:
            preset_name = '*tree'
        preset_operations = OperationsPreset(task=task, preset_name=preset_name)

        if self.api_params.get('available_operations') is not None:
            del self.api_params['available_operations']
        self.api_params = preset_operations.composer_params_based_on_preset(self.api_params, data_type)
        param_dict = {
            'task': self.task,
            'logger': self.log
        }
        self.api_params = {**self.api_params, **param_dict}

    def _parse_input_params(self, input_params: Dict[str, Any]):
        """ Parses input params into different class fields """

        # reset logging level for Singleton
        Log().reset_logging_level(input_params['logging_level'])
        self.log = default_log(prefix='FEDOT logger')

        simple_keys = ['problem', 'n_jobs', 'timeout']
        self.api_params = {k: input_params[k] for k in simple_keys}

        default_evo_params = self.get_default_evo_params(self.api_params['problem'])
        if input_params['composer_tuner_params'] is None:
            evo_params = default_evo_params
        else:
            evo_params = {**default_evo_params, **input_params['composer_tuner_params']}
        self.api_params.update(evo_params)
        if 'preset' not in input_params['composer_tuner_params']:
            self.api_params['preset'] = 'auto'

        # If early_stopping_generations is not specified,
        # than estimate it as in time-based manner as: 0.33 * composing_timeout.
        # The minimal number of generations is 5.
        if 'early_stopping_iterations' not in input_params['composer_tuner_params']:
            if input_params['timeout']:
                depending_on_timeout = int(input_params['timeout']/3)
                self.api_params['early_stopping_iterations'] = \
                    depending_on_timeout if depending_on_timeout > 5 else 5

        specified_seed = input_params['seed']
        if specified_seed is not None:
            np.random.seed(specified_seed)
            random.seed(specified_seed)
            RandomStateHandler.MODEL_FITTING_SEED = specified_seed

        if self.api_params['problem'] == 'ts_forecasting' and input_params['task_params'] is None:
            self.log.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            input_params['task_params'] = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        if self.api_params['problem'] == 'clustering':
            raise ValueError('This type of task is not not supported in API now')

        self.task = self.get_task_params(self.api_params['problem'], input_params['task_params'])
        self.metric_name = self.get_default_metric(self.api_params['problem'])
        self.api_params.pop('problem')

    @staticmethod
    def get_default_evo_params(problem: str):
        """ Dictionary with default parameters for composer """
        params = {'max_depth': 6,
                  'max_arity': 3,
                  'pop_size': 20,
                  'num_of_generations': None,
                  'keep_n_best': 1,
                  'with_tuning': True,
                  'preset': AUTO_PRESET_NAME,
                  'genetic_scheme': None,
                  'early_stopping_iterations': 30,
                  'early_stopping_timeout': 10,
                  'use_pipelines_cache': True,
                  'use_preprocessing_cache': True,
                  'cache_folder': None}

        if problem in ['classification', 'regression']:
            params['cv_folds'] = 5
        elif problem == 'ts_forecasting':
            params['cv_folds'] = 3
            params['validation_blocks'] = 2
        return params

    @staticmethod
    def get_default_metric(problem: str) -> Union[str, List[str]]:
        default_test_metric_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return default_test_metric_dict[problem]

    @staticmethod
    def get_task_params(problem: str, task_params: Optional[TaskParams] = None):
        """ Return task parameters by machine learning problem name (string) """
        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        return task_dict[problem]


def check_timeout_vs_generations(api_params):
    timeout = api_params['timeout']
    num_of_generations = api_params['num_of_generations']
    if timeout in [-1, None]:
        api_params['timeout'] = None
        if num_of_generations is None:
            raise ValueError('"num_of_generations" should be specified if infinite "timeout" is given')
    elif timeout > 0:
        if num_of_generations is None:
            api_params['num_of_generations'] = 10000
    else:
        raise ValueError(f'invalid "timeout" value: timeout={timeout}')
    return api_params
