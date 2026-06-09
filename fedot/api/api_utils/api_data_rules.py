from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from fedot.core.data.common.enums import StateEnum
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class NormalizedFeatures:
    features: Any
    shared_index: Optional[Any]


@dataclass(frozen=True)
class PreprocessingPlan:
    steps: Tuple[str, ...]
    mark_auto_preprocessed: bool


@dataclass(frozen=True)
class PredictionPlan:
    output_mode: Optional[str]
    use_in_sample_forecast: bool
    flatten_prediction: bool
    horizon: Optional[int]


@dataclass(frozen=True)
class StrategyResolution:
    strategy_factory: Any


@dataclass(frozen=True)
class TensorDataDefinitionPlan:
    backend_name: str
    state: StateEnum


@dataclass(frozen=True)
class TensorDataCreationRequest:
    backend_name: str
    spec_kwargs: dict


def build_tensordata_definition_plan(backend_name: str, is_predict: bool) -> TensorDataDefinitionPlan:
    return TensorDataDefinitionPlan(
        backend_name=backend_name,
        state=StateEnum.PREDICT if is_predict else StateEnum.FIT,
    )


class DataDefinitionResolutionError(TypeError):
    pass


_FIT_PREPROCESSING_STEPS = (
    'obligatory_prepare_for_fit',
    'optional_prepare_for_fit',
    'convert_indexes_for_fit',
    'reduce_memory_size',
)

_PREDICT_PREPROCESSING_STEPS = (
    'obligatory_prepare_for_predict',
    'optional_prepare_for_predict',
    'convert_indexes_for_predict',
    'update_indices_for_time_series',
    'reduce_memory_size',
)


def normalize_features_for_definition(features: Any) -> NormalizedFeatures:
    if isinstance(features, dict) and 'idx' in features:
        normalized_features = dict(features)
        shared_index = normalized_features.pop('idx')
        return NormalizedFeatures(features=normalized_features, shared_index=shared_index)
    return NormalizedFeatures(features=features, shared_index=None)


def iter_shared_index_assignments(data: Any, shared_index: Optional[Any]) -> Tuple[Tuple[str, Any], ...]:
    if shared_index is None or not isinstance(data, dict):
        return tuple()
    return tuple((data_source_name, shared_index) for data_source_name in data)


def plan_fit_preprocessing() -> PreprocessingPlan:
    return PreprocessingPlan(steps=_FIT_PREPROCESSING_STEPS, mark_auto_preprocessed=True)


def plan_predict_preprocessing() -> PreprocessingPlan:
    return PreprocessingPlan(steps=_PREDICT_PREPROCESSING_STEPS, mark_auto_preprocessed=True)


def plan_prediction(task_type: TaskTypesEnum,
                    in_sample: bool,
                    validation_blocks: Optional[int],
                    forecast_length: Optional[int]) -> PredictionPlan:
    if task_type == TaskTypesEnum.classification:
        return PredictionPlan(
            output_mode='labels',
            use_in_sample_forecast=False,
            flatten_prediction=False,
            horizon=None,
        )

    if task_type == TaskTypesEnum.ts_forecasting and in_sample:
        blocks = validation_blocks or 1
        horizon = (forecast_length or 0) * blocks
        return PredictionPlan(
            output_mode=None,
            use_in_sample_forecast=True,
            flatten_prediction=False,
            horizon=horizon,
        )

    if task_type == TaskTypesEnum.ts_forecasting:
        return PredictionPlan(
            output_mode=None,
            use_in_sample_forecast=False,
            flatten_prediction=True,
            horizon=None,
        )

    return PredictionPlan(
        output_mode=None,
        use_in_sample_forecast=False,
        flatten_prediction=False,
        horizon=None,
    )


def resolve_strategy(features: Any, strategy_dispatch: Iterable[Tuple[type, Any]]) -> StrategyResolution:
    for source_type, strategy_factory in strategy_dispatch:
        if isinstance(features, source_type):
            return StrategyResolution(strategy_factory=strategy_factory)

    supported_sources = ', '.join(
        source_type.__name__ for source_type, _ in strategy_dispatch)
    raise DataDefinitionResolutionError(
        f'Unsupported features type: {type(features).__name__}. Supported types: {supported_sources}.'
    )
