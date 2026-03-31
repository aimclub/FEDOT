import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.api.main_rules import (
    build_industrial_metrics_plan,
    build_industrial_predict_plan,
    normalize_industrial_prediction,
    trim_industrial_forecast,
)


def test_build_industrial_predict_plan_tracks_solver_mode_and_forecast_tail():
    plan = build_industrial_predict_plan(
        predict_mode='labels',
        solver_is_fedot_class=False,
        solver_is_pipeline_class=True,
        has_target_encoder=True,
        predict_task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=3)),
    )

    assert plan.custom_predict is False
    assert plan.labels_output is True
    assert plan.use_pipeline_predict_mode is True
    assert plan.forecast_length == 3


def test_build_industrial_metrics_plan_detects_mapping_and_encoder_usage():
    dict_plan = build_industrial_metrics_plan(np.array([[1], [0]]), {'a': np.array([1, 0])}, True)
    array_plan = build_industrial_metrics_plan(np.array([[1], [0]]), np.array([1, 0]), True)

    assert dict_plan.prediction_is_mapping is True
    assert dict_plan.use_target_encoder is False
    assert array_plan.prediction_is_mapping is False
    assert array_plan.use_target_encoder is True


def test_normalize_industrial_prediction_unwraps_outputdata_and_trim_forecast_is_optional():
    raw = OutputData(
        idx=np.arange(4),
        predict=np.array([1, 2, 3, 4]),
        target=None,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    normalized = normalize_industrial_prediction(raw)

    assert np.array_equal(normalized, np.array([1, 2, 3, 4]))
    assert np.array_equal(trim_industrial_forecast(normalized, None), normalized)
    assert np.array_equal(trim_industrial_forecast(normalized, 2), np.array([3, 4]))
