from dataclasses import dataclass
from typing import Any, Optional

from fedot.api.api_utils.schemas import TensorMetricsExecutionSchema
from fedot.core.data.tensor_data import TensorData
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


@dataclass(frozen=True)
class TuneExecutionPlan:
    tensor_data: Any
    cv_folds: Optional[int]
    n_jobs: int
    metric: Any

@dataclass(frozen=True)
class PredictExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class PredictProbaExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class FitExecutionPlan:
    fit_method_name: str


@dataclass(frozen=True)
class ForecastExecutionPlan:
    horizon: int
    clear_target: bool


@dataclass(frozen=True)
class MetricsExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class MetricsValidationPlan:
    metrics: Any
    metric_names: list[str]
    in_sample: bool
    validation_blocks: Optional[int]
    rounding_order: int


@dataclass(frozen=True)
class ExplainExecutionPlan:
    method: str
    visualization: bool


def build_predict_plan(output_mode: str = 'default') -> PredictExecutionPlan:
    return PredictExecutionPlan(output_mode=output_mode)


def build_predict_proba_plan(probs_for_all_classes: bool) -> PredictProbaExecutionPlan:
    return PredictProbaExecutionPlan(output_mode=resolve_predict_proba_mode(probs_for_all_classes))


def build_forecast_plan(
        requested_horizon: Optional[int],
        forecast_length: int) -> ForecastExecutionPlan:
    return ForecastExecutionPlan(
        horizon=resolve_forecast_horizon(requested_horizon, forecast_length),
        clear_target=True,
    )


def build_metrics_plan() -> MetricsExecutionPlan:
    return MetricsExecutionPlan(output_mode='default')


def build_metrics_validation_plan(
    is_pipeline_fitted: bool,
    metric_names: Any,
    default_metrics: Any,
    requested_in_sample: Optional[bool],
    default_in_sample: bool,
    validation_blocks: Optional[int],
    rounding_order: int,
    context: ValidationContext = None,
) -> MetricsValidationPlan:
    validated = load_validated(
        TensorMetricsExecutionSchema(),
        {
            'is_pipeline_fitted': is_pipeline_fitted,
            'metric_names': metric_names,
            'default_metrics': default_metrics,
            'requested_in_sample': requested_in_sample,
            'default_in_sample': default_in_sample,
            'validation_blocks': validation_blocks,
            'rounding_order': rounding_order,
        },
        context,
        prefix='tensor_metrics',
    )

    metrics = validated['metric_names'] if validated['metric_names'] else validated['default_metrics']
    if isinstance(metrics, (str, bytes)):
        metrics = [metrics]
    else:
        try:
            metrics = list(metrics)
        except TypeError:
            metrics = [metrics]

    in_sample = validated['requested_in_sample']
    if in_sample is None:
        in_sample = validated['default_in_sample']

    resolved_validation_blocks = validated['validation_blocks'] if in_sample else None

    return MetricsValidationPlan(
        metrics=metrics,
        metric_names=[str(metric) for metric in metrics],
        in_sample=in_sample,
        validation_blocks=resolved_validation_blocks,
        rounding_order=validated['rounding_order'],
    )


def build_explain_plan(method: str, visualization: bool) -> ExplainExecutionPlan:
    return ExplainExecutionPlan(method=method, visualization=visualization)


def build_tune_execution_plan(
    tensor_data: Optional[TensorData],
    train_data: TensorData,
    requested_cv_folds: Optional[int],
    default_cv_folds: Optional[int],
    requested_n_jobs: Optional[int],
    default_n_jobs: int,
    requested_metric: Any,
    default_metric: Any
) -> TuneExecutionPlan:
    resolved_tensor_data = train_data if tensor_data is None else tensor_data
    resolved_cv_folds = default_cv_folds if requested_cv_folds is None else requested_cv_folds
    resolved_n_jobs = default_n_jobs if requested_n_jobs is None else requested_n_jobs
    resolved_metric = default_metric if requested_metric is None else requested_metric
    return TuneExecutionPlan(
        tensor_data=resolved_tensor_data,
        cv_folds=resolved_cv_folds,
        n_jobs=resolved_n_jobs,
        metric=resolved_metric,
    )


def resolve_predict_proba_mode(probs_for_all_classes: bool) -> str:
    return 'full_probs' if probs_for_all_classes else 'probs'


def resolve_forecast_horizon(requested_horizon: Optional[int], forecast_length: int) -> int:
    return forecast_length if requested_horizon is None else requested_horizon
