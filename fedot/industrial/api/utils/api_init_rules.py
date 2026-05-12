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


@dataclass(frozen=True)
class ComputationalConfigPlan:
    backend: str
    distributed: dict
    output_folder: Optional[str]
    cache: Optional[dict]
    automl_folder: Optional[str]


@dataclass(frozen=True)
class AutomlConfigPlan:
    task: Optional[str]
    task_params: Optional[dict]
    initial_assumption: object
    use_automl: bool
    available_operations: list
    optimisation_strategy: Optional[dict]


def build_industrial_context_plan(problem: str,
                                  strategy: Optional[str] = None,
                                  task_params: Optional[dict] = None,
                                  regression_tasks=None) -> IndustrialContextPlan:
    strategy_name = strategy or 'default'
    regression_task_names = regression_tasks or [
        'ts_forecasting', 'regression']
    raw_task_params = task_params or {}
    is_default_fedot_context = 'tabular' in strategy_name
    is_regression_task_context = problem in regression_task_names
    is_forecasting_context = problem == 'ts_forecasting' and bool(
        raw_task_params)
    normalized_task_params = None
    if is_forecasting_context:
        normalized_task_params = TsForecastingParams(
            forecast_length=raw_task_params['forecast_length'])
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


def build_computational_config_plan(backend: str = 'cpu',
                                    distributed: Optional[dict] = None,
                                    output_folder: Optional[str] = None,
                                    cache_dict: Optional[dict] = None,
                                    automl_folder: Optional[str] = None,
                                    default_dask_params: Optional[dict] = None) -> ComputationalConfigPlan:
    normalized_default_dask = dict(default_dask_params or {})
    normalized_distributed = dict(
        distributed) if distributed is not None else normalized_default_dask
    normalized_cache = dict(cache_dict) if cache_dict is not None else None
    return ComputationalConfigPlan(
        backend=backend,
        distributed=normalized_distributed,
        output_folder=output_folder,
        cache=normalized_cache,
        automl_folder=automl_folder,
    )


def build_automl_config_plan(task: Optional[str] = None,
                             task_params: Optional[dict] = None,
                             initial_assumption=None,
                             use_automl: bool = False,
                             available_operations=None,
                             optimisation_strategy: Optional[dict] = None,
                             default_available_operations_factory: Optional[Callable[[str],
                                                                                     list]] = None) -> AutomlConfigPlan:
    normalized_task_params = dict(
        task_params) if task_params is not None else None
    normalized_available_operations = list(
        available_operations) if available_operations is not None else None
    if normalized_available_operations is None and task is not None and default_available_operations_factory is not None:
        normalized_available_operations = list(
            default_available_operations_factory(task))
    return AutomlConfigPlan(
        task=task,
        task_params=normalized_task_params,
        initial_assumption=initial_assumption,
        use_automl=use_automl,
        available_operations=normalized_available_operations,
        optimisation_strategy=optimisation_strategy,
    )
