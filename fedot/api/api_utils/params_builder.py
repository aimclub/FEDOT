import random
from typing import Optional

import numpy as np

from fedot.core.constants import DEFAULT_FORECAST_LENGTH
from fedot.core.log import Log, default_log
from fedot.core.optimisers.gp_comp.evaluation import determine_n_jobs
from fedot.core.repository.tasks import TaskParams, TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utilities.random import RandomStateHandler


class ApiParamsBuilder:
    def __init__(self, problem: str, task_params: Optional[TaskParams] = None):
        self.task = self._get_task_with_params(problem, task_params)
        self.seed = None
        self.n_jobs = -1
        self.params = {}
        self.log = default_log(prefix='FEDOT logger')
        self.timeout = 5
        self.parallelization_mode = 'populational'

    def _get_task_with_params(self, problem: str, task_params: Optional[TaskParams] = None):
        if problem == 'ts_forecasting' and task_params is None:
            self.log.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            task_params = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        if problem == 'clustering':
            raise ValueError('This type of task is not not supported in API now')

        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'clustering': Task(TaskTypesEnum.clustering, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        return task_dict[problem]

    def with_logging_level(self, logging_level: int):
        # reset logging level for Singleton
        Log().reset_logging_level(logging_level)
        self.log = default_log(prefix='FEDOT logger')
        return self

    def with_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        RandomStateHandler.MODEL_FITTING_SEED = seed
        self.seed = seed
        return self

    def with_n_jobs(self, n_jobs: int):
        self.n_jobs = determine_n_jobs(n_jobs)
        return self

    def with_composer_tuner_params(self, params: dict):
        self.params = params
        return self

    def with_timeout(self, timeout: int):
        self.timeout = timeout
        return self

    def with_parallelization_mode(self, parallelization_mode: str):
        self.parallelization_mode = parallelization_mode
        return self

    def build(self):
        pass
