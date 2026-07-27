from dataclasses import dataclass
from typing import Any, Dict, Optional

from fedot.core.constants import AUTO_PRESET_NAME, DEFAULT_FORECAST_LENGTH
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum, TsForecastingParams
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotValidationError
from fedot.api.api_utils.schemas import validate_problem, validate_timeout_generations


@dataclass(frozen=True)
class TaskResolution:
    task: Task
    warning_message: Optional[str]


@dataclass(frozen=True)
class TimeoutResolution:
    timeout: Optional[float]
    num_of_generations: Optional[int]


_SUPPORTED_PROBLEMS = {
    'regression': TaskTypesEnum.regression,
    'classification': TaskTypesEnum.classification,
    'ts_forecasting': TaskTypesEnum.ts_forecasting,
}


def resolve_task(problem: str,
                 task_params: Optional[TaskParams],
                 default_forecast_length: int = DEFAULT_FORECAST_LENGTH,
                 context: Optional[ValidationContext] = None) -> TaskResolution:
    try:
        validate_problem(problem, context)
    except FedotValidationError as exc:
        raise FedotValidationError(
            f'Wrong type name of the given task: {problem}',
            field_name='problem',
        ) from exc

    warning_message = None
    resolved_task_params = task_params
    if problem == 'ts_forecasting' and task_params is None:
        warning_message = f'The value of the forecast depth was set to {default_forecast_length}.'
        resolved_task_params = TsForecastingParams(
            forecast_length=default_forecast_length)

    task_type = _SUPPORTED_PROBLEMS[problem]
    return TaskResolution(task=Task(task_type, task_params=resolved_task_params), warning_message=warning_message)


def normalize_timeout_and_generations(timeout: Optional[float],
                                      num_of_generations: Optional[int],
                                      context: Optional[ValidationContext] = None) -> TimeoutResolution:
    validated = validate_timeout_generations(timeout, num_of_generations, context)

    if timeout in (-1, None):
        return TimeoutResolution(timeout=None, num_of_generations=validated['num_of_generations'])

    if num_of_generations is None:
        return TimeoutResolution(timeout=timeout, num_of_generations=10000)
    return TimeoutResolution(timeout=timeout, num_of_generations=num_of_generations)


def build_label_encoded_preset_name(current_preset: Optional[str]) -> str:
    if current_preset:
        return f'{current_preset}*tree'
    return '*tree'


def should_update_available_operations(preset: Optional[str]) -> bool:
    return preset != AUTO_PRESET_NAME


def merge_param_recommendations(current_params: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, Any]:
    updated_params = dict(current_params)
    for key, value in recommendations.items():
        updated_params[key] = value
    return updated_params
