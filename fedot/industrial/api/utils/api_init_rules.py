from dataclasses import dataclass
from typing import Callable, Optional, Union

from fedot.core.repository.tasks import TsForecastingParams


@dataclass(frozen=True)
class IndustrialContextPlan:
    strategy_name: str
    is_default_fedot_context: bool
    is_regression_task_context: bool
    is_forecasting_context: bool
    normalized_task_params: Optional[TsForecastingParams]


@dataclass(frozen=True)
class LearningLossPlan:
    quality_loss: Optional[Union[Callable, str]]
    computational_loss: Optional[Union[Callable, str]]
    structural_loss: Optional[Union[Callable, str]]


@dataclass(frozen=True)
class ApiManagerStatePlan:
    solver: object = None
    predicted_labels: object = None
    predicted_probs: object = None
    predict_data: object = None
    dask_client: object = None
    dask_cluster: object = None
    target_encoder: object = None
    is_finetuned: bool = False


def build_industrial_context_plan(problem: str,
                                  strategy: Optional[str] = None,
                                  task_params: Optional[dict] = None,
                                  regression_tasks=None) -> IndustrialContextPlan:
    strategy_name = strategy or 'default'
    regression_task_names = regression_tasks or ['ts_forecasting', 'regression']
    raw_task_params = task_params or {}
    is_default_fedot_context = 'tabular' in strategy_name
    is_regression_task_context = problem in regression_task_names
    is_forecasting_context = problem == 'ts_forecasting' and bool(raw_task_params)
    normalized_task_params = None
    if is_forecasting_context:
        normalized_task_params = TsForecastingParams(forecast_length=raw_task_params['forecast_length'])
    return IndustrialContextPlan(
        strategy_name=strategy_name,
        is_default_fedot_context=is_default_fedot_context,
        is_regression_task_context=is_regression_task_context,
        is_forecasting_context=is_forecasting_context,
        normalized_task_params=normalized_task_params,
    )


def resolve_initial_assumption_problem(problem: str, strategy_name: str, is_default_fedot_context: bool) -> str:
    return f'{problem}_{strategy_name}' if is_default_fedot_context else problem


def build_learning_loss_plan(loss: Union[Callable, str, dict, None]) -> LearningLossPlan:
    if isinstance(loss, dict):
        return LearningLossPlan(
            quality_loss=loss.get('quality_loss'),
            computational_loss=loss.get('computational_loss'),
            structural_loss=loss.get('structural_loss'),
        )
    if callable(loss):
        return LearningLossPlan(quality_loss=loss, computational_loss=None, structural_loss=None)
    return LearningLossPlan(quality_loss=None, computational_loss=None, structural_loss=None)


def build_api_manager_state_plan() -> ApiManagerStatePlan:
    return ApiManagerStatePlan()
