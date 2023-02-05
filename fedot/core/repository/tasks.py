from dataclasses import dataclass
from typing import Any, List, Optional

from fedot.core.utilities.data_structures import ComparableEnum as Enum


@dataclass
class TaskParams:
    pass


@dataclass
class TsForecastingParams(TaskParams):
    forecast_length: int

    def __post_init__(self):
        if self.forecast_length < 1:
            raise ValueError('Forecast length should be more then 0')


class TaskTypesEnum(Enum):
    classification = 'classification'
    regression = 'regression'
    ts_forecasting = 'ts_forecasting'
    clustering = 'clustering'  # not applicable as main task yet

class PresetTypesEnum(Enum):
```preset: name of preset for model building (e.g. 'best_quality', 'fast_train', 'gpu'):
            .. details:: possible ``preset`` options:
                - ``best_quality`` -> All models that are available for this data type and task are used
                - ``fast_train`` -> Models that learn quickly. This includes preprocessing operations
                  (data operations) that only reduce the dimensionality of the data, but cannot increase it.
                  For example, there are no polynomial features and one-hot encoding operations
                - ``stable`` -> The most reliable preset in which the most stable operations are included.
                - ``auto`` -> Automatically determine which preset should be used.
                - ``gpu`` -> Models that use GPU resources for computation.
                - ``ts`` -> A special preset with models for time series forecasting task.
                - ``automl`` -> A special preset with only AutoML libraries such as TPOT and H2O as operations.```
    best_quality = 'best_quality'
    fast_train = 'fast_train'
    stable = 'stable'
    auto = 'auto'
    gpu = 'gpu'
    ts = 'ts'
    automl = 'automl'
    



@dataclass
class Task:
    task_type: TaskTypesEnum
    task_params: Optional[TaskParams] = None


# local tasks that can be solved as a part of global tasks
def compatible_task_types(main_task_type: TaskTypesEnum) -> List[TaskTypesEnum]:
    _compatible_task_types = {
        TaskTypesEnum.ts_forecasting: [TaskTypesEnum.regression],
        TaskTypesEnum.classification: [TaskTypesEnum.clustering],
        TaskTypesEnum.regression: [TaskTypesEnum.clustering, TaskTypesEnum.classification]
    }
    if main_task_type not in _compatible_task_types:
        return []
    return _compatible_task_types[main_task_type]


def extract_task_param(task: Task) -> Any:
    try:
        task_params = task.task_params
        if isinstance(task_params, TsForecastingParams):
            prediction_len = task_params.forecast_length
            return prediction_len
        else:
            raise ValueError('Incorrect parameters type for data')
    except AttributeError as ex:
        raise AttributeError(f'Params are required for the {task} task: {ex}')
