from dataclasses import dataclass
from typing import Optional, Tuple

from fedot.core.data.data import OutputData


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
