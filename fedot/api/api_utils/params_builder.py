from typing import Optional

from golem.core.log import default_log
from golem.core.optimisers.genetic.evaluation import determine_n_jobs

from fedot.api.api_utils.params import ApiParams
from fedot.core.constants import DEFAULT_FORECAST_LENGTH
from fedot.core.repository.tasks import TaskParams, TsForecastingParams, Task, TaskTypesEnum


class ApiParamsBuilder:
    def __init__(self, problem: str, task_params: Optional[TaskParams] = None):
        self.task = self._get_task_with_params(problem, task_params)
        self.n_jobs = -1
        self.params = {}
        self.timeout = 5
        self.log = default_log(self)

    def _get_task_with_params(self, problem: str, task_params: Optional[TaskParams] = None):
        if problem == 'ts_forecasting' and task_params is None:
            self.log.warning(f'The value of the forecast depth was set to {DEFAULT_FORECAST_LENGTH}.')
            task_params = TsForecastingParams(forecast_length=DEFAULT_FORECAST_LENGTH)

        task_dict = {'regression': Task(TaskTypesEnum.regression, task_params=task_params),
                     'classification': Task(TaskTypesEnum.classification, task_params=task_params),
                     'ts_forecasting': Task(TaskTypesEnum.ts_forecasting, task_params=task_params)
                     }
        try:
            return task_dict[problem]
        except ValueError as exc:
            ValueError('Wrong type name of the given task')

    def with_n_jobs(self, n_jobs: int):
        self.n_jobs = determine_n_jobs(n_jobs)
        return self

    def with_composer_tuner_params(self, params: dict):
        self.params = params
        return self

    def with_timeout(self, timeout: float):
        self.timeout = timeout
        return self

    def build(self):
        api_params = ApiParams(input_params=self.params, n_jobs=self.n_jobs,
                               timeout=self.timeout, task=self.task)
        return api_params
