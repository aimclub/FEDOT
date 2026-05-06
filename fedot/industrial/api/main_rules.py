from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Tuple

from fedot.core.data.input_data.data import OutputData
from fedot.industrial.core.repository.constanst_repository import (
    FEDOT_TUNER_STRATEGY,
    FEDOT_TUNING_METRICS,
)


@dataclass(frozen=True)
class IndustrialPredictPlan:
    custom_predict: bool
    labels_output: bool
    use_pipeline_predict_mode: bool
    forecast_length: Optional[int]


@dataclass(frozen=True)
class IndustrialMetricsPlan:
    valid_shape: Tuple[int, ...]
    prediction_is_mapping: bool
    use_target_encoder: bool


@dataclass(frozen=True)
class IndustrialFitPlan:
    use_solver_fit: bool


@dataclass(frozen=True)
class IndustrialPredictProbaPlan:
    normalized_mode: str


@dataclass(frozen=True)
class IndustrialFinetunePlan:
    should_process_input: bool
    normalized_tuning_params: dict


@dataclass(frozen=True)
class IndustrialMetricsRequestPlan:
    warn_missing_probabilities: bool


def build_industrial_predict_plan(predict_mode: str,
                                  solver_is_fedot_class: bool,
                                  solver_is_pipeline_class: bool,
                                  has_target_encoder: bool,
                                  predict_task) -> IndustrialPredictPlan:
    custom_predict = not solver_is_fedot_class and not solver_is_pipeline_class
    forecast_length = None
    if getattr(predict_task.task_type, 'value', '').__contains__('forecasting'):
        forecast_length = predict_task.task_params.forecast_length
    return IndustrialPredictPlan(
        custom_predict=custom_predict,
        labels_output=predict_mode in ['labels'],
        use_pipeline_predict_mode=solver_is_pipeline_class,
        forecast_length=forecast_length,
    )


def normalize_industrial_prediction(raw_prediction):
    return raw_prediction.predict if isinstance(raw_prediction, OutputData) else raw_prediction


def trim_industrial_forecast(prediction, forecast_length: Optional[int]):
    if forecast_length is None:
        return prediction
    return prediction[-forecast_length:]


def build_industrial_metrics_plan(target, predicted_labels, has_target_encoder: bool) -> IndustrialMetricsPlan:
    return IndustrialMetricsPlan(
        valid_shape=target.shape,
        prediction_is_mapping=isinstance(predicted_labels, dict),
        use_target_encoder=has_target_encoder and not isinstance(predicted_labels, dict),
    )


def build_industrial_fit_plan(strategy) -> IndustrialFitPlan:
    return IndustrialFitPlan(use_solver_fit=not isinstance(strategy, Callable))


def build_industrial_predict_proba_plan(predict_mode: str,
                                        is_regression_task_context: bool) -> IndustrialPredictProbaPlan:
    normalized_mode = 'labels' if is_regression_task_context else predict_mode
    return IndustrialPredictProbaPlan(normalized_mode=normalized_mode)


def build_industrial_finetune_plan(is_fedot_datatype: bool,
                                   task_name: str,
                                   tuning_params: Optional[dict]) -> IndustrialFinetunePlan:
    normalized_tuning_params = dict(tuning_params or {})
    tuner_name = normalized_tuning_params.get('tuner', 'sequential')
    normalized_tuning_params['metric'] = FEDOT_TUNING_METRICS[task_name]
    normalized_tuning_params['tuner'] = FEDOT_TUNER_STRATEGY[tuner_name]
    return IndustrialFinetunePlan(
        should_process_input=not is_fedot_datatype,
        normalized_tuning_params=normalized_tuning_params,
    )


def build_industrial_metrics_request_plan(problem: str,
                                          probs,
                                          metric_names: Optional[tuple]) -> IndustrialMetricsRequestPlan:
    warn_missing_probabilities = all([
        problem == 'classification',
        probs is None,
        metric_names is not None,
        'roc_auc' in metric_names,
    ])
    return IndustrialMetricsRequestPlan(warn_missing_probabilities=warn_missing_probabilities)


@dataclass(frozen=True)
class IndustrialSavePlan:
    selected_mode: str
    save_all: bool
    include_opt_hist: bool


@dataclass(frozen=True)
class IndustrialLoadPlan:
    resolved_path: str
    load_multiple_pipelines: bool


@dataclass(frozen=True)
class IndustrialExplainPlan:
    metric: str
    window: int
    samples: int
    threshold: int
    name: str
    method: str


@dataclass(frozen=True)
class IndustrialHistoryVisualizationPlan:
    selected_mode: str
    visualize_all: bool


def build_industrial_save_plan(mode: str, is_fedot_solver: bool) -> IndustrialSavePlan:
    return IndustrialSavePlan(
        selected_mode=mode,
        save_all='all' in mode,
        include_opt_hist=is_fedot_solver,
    )


def build_industrial_load_plan(path: str, dir_list) -> IndustrialLoadPlan:
    resolved_path = path
    if 'pipeline_saved' not in path:
        saved_pipes = [entry for entry in dir_list if 'pipeline_saved' in entry]
        if saved_pipes:
            resolved_path = f'{path}/{saved_pipes[0]}'
    return IndustrialLoadPlan(
        resolved_path=resolved_path,
        load_multiple_pipelines=('fitted_operations' in dir_list and any(
            'pipeline_saved' in entry for entry in dir_list
        )),
    )


def build_industrial_explain_plan(explaining_config: dict) -> IndustrialExplainPlan:
    return IndustrialExplainPlan(
        metric=explaining_config.get('metric', 'rmse'),
        window=explaining_config.get('window', 5),
        samples=explaining_config.get('samples', 1),
        threshold=explaining_config.get('threshold', 90),
        name=explaining_config.get('name', 'test'),
        method=explaining_config.get('method', 'point'),
    )


def build_industrial_history_visualization_plan(mode: str) -> IndustrialHistoryVisualizationPlan:
    return IndustrialHistoryVisualizationPlan(
        selected_mode=mode,
        visualize_all=mode == 'all',
    )
