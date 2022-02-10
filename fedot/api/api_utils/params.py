import random
from typing import Optional, Dict, Union

import numpy as np

from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams, TaskParams


class ApiParams:

    def __init__(self):
        self.default_forecast_length = 30
        self.api_params = None
        self.log = None
        self.task = None
        self.metric_to_compose = None
        self.task_params = None
        self.metric_name = None
        self.initial_assumption = None

    def check_input_params(self, **input_params):
        self.metric_to_compose = None
        self.api_params['problem'] = input_params['problem']
        self.log = default_log('FEDOT logger', verbose_level=input_params['verbose_level'])

        if input_params['seed'] is not None:
            np.random.seed(input_params['seed'])
            random.seed(input_params['seed'])

        if 'metric' in self.api_params:
            self.api_params['composer_metric'] = self.api_params['metric']
            del self.api_params['metric']
            self.metric_to_compose = self.api_params['composer_metric']

        if input_params['problem'] == 'ts_forecasting' and input_params['task_params'] is None:
            self.log.warn('The value of the forecast depth was set to {}.'.format(self.default_forecast_length))
            self.task_params = TsForecastingParams(forecast_length=self.default_forecast_length)

        if input_params['problem'] == 'clustering':
            raise ValueError('This type of task is not not supported in API now')

    def get_initial_params(self, **input_params):
        if input_params['composer_params'] is None:
            self.api_params = self.get_default_evo_params(problem=input_params['problem'])
        else:
            self.api_params = {**self.get_default_evo_params(problem=input_params['problem']),
                               **input_params['composer_params']}

        self.check_input_params(**input_params)

        self.task = self.get_task_params(input_params['problem'],
                                         input_params['task_params'])
        self.metric_name = self.get_default_metric(input_params['problem'])

        param_dict = {
            'task': self.task,
            'logger': self.log,
            'metric_name': self.metric_name,
            'composer_metric': self.metric_to_compose
        }
        self.api_params = {**self.api_params, **param_dict}

    def initialize_params(self, **input_params):
        self.get_initial_params(**input_params)
        preset_operations = OperationsPreset(task=self.task, preset_name=input_params['preset'])
        self.api_params = preset_operations.composer_params_based_on_preset(composer_params=self.api_params)

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
                self.change_preset_for_label_encoded_data(input_data.task)

    def change_preset_for_label_encoded_data(self, task: Task):
        """ Change preset on tree like preset, if data had been label encoded """
        if self.api_params.get('preset') is not None:
            preset_name = ''.join((self.api_params['preset'], '*tree'))
        else:
            preset_name = '*tree'
        preset_operations = OperationsPreset(task=task, preset_name=preset_name)
        del self.api_params['available_operations']
        self.api_params = preset_operations.composer_params_based_on_preset(composer_params=self.api_params)
        param_dict = {
            'task': self.task,
            'logger': self.log,
            'metric_name': self.metric_name,
            'composer_metric': self.metric_to_compose
        }
        self.api_params = {**self.api_params, **param_dict}

    @staticmethod
    def get_default_evo_params(problem: str):
        """ Dictionary with default parameters for composer """
        params = {'max_depth': 3,
                  'max_arity': 4,
                  'pop_size': 20,
                  'num_of_generations': 20,
                  'timeout': 2,
                  'with_tuning': True,
                  'preset': 'best_quality',
                  'genetic_scheme': None,
                  'history_folder': None,
                  'stopping_after_n_generation': 10}

        if problem in ['classification', 'regression']:
            params['cv_folds'] = 3
        elif problem == 'ts_forecasting':
            params['cv_folds'] = 3
            params['validation_blocks'] = 2
        return params

    @staticmethod
    def get_default_metric(problem: str):
        default_test_metric_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return default_test_metric_dict[problem]

    @staticmethod
    def get_task_params(problem, task_params: Optional[TaskParams] = None):
        """ Return task parameters by machine learning problem name (string) """
        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        return task_dict[problem]
