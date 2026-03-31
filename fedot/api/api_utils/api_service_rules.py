from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class TuneExecutionPlan:
    input_data: Any
    cv_folds: Optional[int]
    n_jobs: int
    metric: Any


@dataclass(frozen=True)
class TensorPredictExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class TensorPredictProbaExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class TensorFitExecutionPlan:
    fit_method_name: str


@dataclass(frozen=True)
class TensorTuneExecutionPlan:
    input_data: Any
    use_tensor_runtime: bool
    builder_method_name: str
    refit_method_name: str


@dataclass(frozen=True)
class TensorForecastExecutionPlan:
    horizon: int
    clear_target: bool


@dataclass(frozen=True)
class TensorMetricsExecutionPlan:
    output_mode: str


@dataclass(frozen=True)
class TensorExplainExecutionPlan:
    method: str
    visualization: bool


def build_tensordata_fit_plan(predefined_model: Any) -> TensorFitExecutionPlan:
    if predefined_model is None:
        raise ValueError('TensorData fit currently supports only predefined models or pipelines.')
    if predefined_model == 'auto':
        raise ValueError(
            'TensorData fit does not support auto assumption generation yet. '
            'Pass a model name or Pipeline.'
        )
    return TensorFitExecutionPlan(fit_method_name='fit_tensordata')


def build_tensordata_predict_plan(output_mode: str = 'default') -> TensorPredictExecutionPlan:
    return TensorPredictExecutionPlan(output_mode=output_mode)


def build_tensordata_predict_proba_plan(probs_for_all_classes: bool) -> TensorPredictProbaExecutionPlan:
    return TensorPredictProbaExecutionPlan(output_mode=resolve_predict_proba_mode(probs_for_all_classes))


def build_tensordata_tune_plan(converted_input_data: Optional[Any], has_tensor_data: bool) -> TensorTuneExecutionPlan:
    return TensorTuneExecutionPlan(
        input_data=converted_input_data,
        use_tensor_runtime=has_tensor_data,
        builder_method_name='build_tensordata' if has_tensor_data else 'build',
        refit_method_name='fit_tensordata' if has_tensor_data else 'fit',
    )


def build_tensordata_forecast_plan(requested_horizon: Optional[int], forecast_length: int) -> TensorForecastExecutionPlan:
    return TensorForecastExecutionPlan(
        horizon=resolve_forecast_horizon(requested_horizon, forecast_length),
        clear_target=True,
    )


def build_tensordata_metrics_plan() -> TensorMetricsExecutionPlan:
    return TensorMetricsExecutionPlan(output_mode='default')


def build_tensordata_explain_plan(method: str, visualization: bool) -> TensorExplainExecutionPlan:
    return TensorExplainExecutionPlan(method=method, visualization=visualization)


def build_tune_execution_plan(input_data: Any,
                              train_data: Any,
                              requested_cv_folds: Optional[int],
                              default_cv_folds: Optional[int],
                              requested_n_jobs: Optional[int],
                              default_n_jobs: int,
                              requested_metric: Any,
                              default_metric: Any) -> TuneExecutionPlan:
    resolved_input_data = train_data if input_data is None else input_data
    resolved_cv_folds = default_cv_folds if requested_cv_folds is None else requested_cv_folds
    resolved_n_jobs = default_n_jobs if requested_n_jobs is None else requested_n_jobs
    resolved_metric = default_metric if requested_metric is None else requested_metric
    return TuneExecutionPlan(
        input_data=resolved_input_data,
        cv_folds=resolved_cv_folds,
        n_jobs=resolved_n_jobs,
        metric=resolved_metric,
    )


def resolve_predict_proba_mode(probs_for_all_classes: bool) -> str:
    return 'full_probs' if probs_for_all_classes else 'probs'


def resolve_forecast_horizon(requested_horizon: Optional[int], forecast_length: int) -> int:
    return forecast_length if requested_horizon is None else requested_horizon
