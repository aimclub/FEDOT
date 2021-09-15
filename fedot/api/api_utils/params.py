import random
import numpy as np

from fedot.core.log import default_log
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.api.api_utils.presets import ApiPresetHelper


class ApiParamsHelper:

    def __init__(self):
        self.default_forecast_length = 30
        return

    def get_default_evo_params(self, problem: str):
        """ Dictionary with default parameters for composer """
        params = {'max_depth': 3,
                  'max_arity': 4,
                  'pop_size': 20,
                  'num_of_generations': 20,
                  'timeout': 2,
                  'with_tuning': False,
                  'preset': 'light_tun',
                  'genetic_scheme': None,
                  'history_folder': None}

        if problem in ['classification', 'regression']:
            params['cv_folds'] = 3
        return params

    def get_default_metric(self, problem: str):
        default_test_metric_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return default_test_metric_dict[problem]

    def get_task_params(self,
                        problem,
                        task_params):
        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        return task_dict[problem]

    def check_input_params(self, **input_params):
        self.metric_to_compose = None
        self.api_params['problem'] = input_params['problem']
        self.log = default_log('FEDOT logger', verbose_level=input_params['verbose_level'])

        if input_params['seed'] is not None:
            np.random.seed(input_params['seed'])
            random.seed(input_params['seed'])

        if input_params['timeout'] is not None:
            self.api_params['timeout'] = self.api_params['timeout']
            self.api_params['num_of_generations'] = 10000

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

        return

    def initialize_params(self, **input_params):
        self.get_initial_params(**input_params)
        preset_model = ApiPresetHelper()
        self.api_params = preset_model.get_preset(task=self.task,
                                                  preset=input_params['preset'],
                                                  composer_params=self.api_params)
        param_dict = {
            'task': self.task,
            'logger': self.log,
            'metric_name': self.metric_name,
            'composer_metric': self.metric_to_compose
        }

        return {**param_dict, **self.api_params}
