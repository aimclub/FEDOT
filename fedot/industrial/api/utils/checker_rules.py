from dataclasses import dataclass
from typing import Any, Optional

from fedot.core.data.common.compatibility_rules import build_data_type_compatibility
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.core.repository.constanst_repository import FEDOT_DATA_TYPE


@dataclass(frozen=True)
class IndustrialDataTypePlan:
    requested_name: str
    fedot_data_type: DataTypesEnum
    tensor_canonical_data_type: DataTypesEnum
    input_compatible_data_type: DataTypesEnum


@dataclass(frozen=True)
class IndustrialTaskPlan:
    task: Task
    have_predict_horizon: bool


@dataclass(frozen=True)
class IndustrialLearningStrategyFlags:
    strategy_name: Optional[str]
    is_big_data: bool
    is_default_fedot_context: bool


def resolve_industrial_data_type_plan(strategy_params: Optional[dict]) -> IndustrialDataTypePlan:
    requested_name = 'tensor' if strategy_params is None else strategy_params.get('data_type', 'tensor')
    if requested_name not in FEDOT_DATA_TYPE:
        raise ValueError(f'Unsupported industrial data_type: {requested_name}')
    fedot_data_type = FEDOT_DATA_TYPE[requested_name]
    compatibility = build_data_type_compatibility(fedot_data_type)
    return IndustrialDataTypePlan(
        requested_name=requested_name,
        fedot_data_type=fedot_data_type,
        tensor_canonical_data_type=compatibility.tensor_canonical,
        input_compatible_data_type=compatibility.input_compatible,
    )


def should_use_predict_horizon(strategy_params: Optional[dict]) -> bool:
    return bool(
        strategy_params is not None
        and strategy_params.get('data_type') == 'time_series'
        and 'detection_window' in strategy_params
    )


def build_industrial_task_plan(task_name: str, strategy_params: Optional[dict]) -> IndustrialTaskPlan:
    have_predict_horizon = should_use_predict_horizon(strategy_params)
    if have_predict_horizon:
        detection_window = strategy_params['detection_window']
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=detection_window))
    elif task_name == 'classification':
        task = Task(TaskTypesEnum.classification)
    elif task_name == 'regression':
        task = Task(TaskTypesEnum.regression)
    elif task_name == 'ts_forecasting':
        forecast_length = None if strategy_params is None else strategy_params.get('forecast_length')
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length or 1))
    else:
        raise ValueError(f'Unsupported industrial task: {task_name}')
    return IndustrialTaskPlan(task=task, have_predict_horizon=have_predict_horizon)


def resolve_learning_strategy_flags(strategy_params: Optional[dict]) -> IndustrialLearningStrategyFlags:
    strategy_name = None if strategy_params is None else strategy_params.get('learning_strategy')
    return IndustrialLearningStrategyFlags(
        strategy_name=strategy_name,
        is_big_data=bool(strategy_name and 'big' in strategy_name),
        is_default_fedot_context=bool(strategy_name and 'tabular' in strategy_name),
    )
