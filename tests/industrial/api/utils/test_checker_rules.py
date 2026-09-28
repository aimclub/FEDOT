import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.industrial.api.utils.checker_rules import (
    build_industrial_task_plan,
    resolve_industrial_data_type_plan,
    resolve_learning_strategy_flags,
    should_use_predict_horizon,
)


def test_resolve_industrial_data_type_plan_builds_fedot_and_tensor_views():
    plan = resolve_industrial_data_type_plan({'data_type': 'time_series'})

    assert plan.requested_name == 'time_series'
    assert plan.fedot_data_type == DataTypesEnum.ts
    assert plan.tensor_canonical_data_type == DataTypesEnum.ts
    assert plan.input_compatible_data_type == DataTypesEnum.ts


def test_resolve_industrial_data_type_plan_uses_tensor_default():
    plan = resolve_industrial_data_type_plan(None)

    assert plan.requested_name == 'tensor'
    assert plan.fedot_data_type == DataTypesEnum.image
    assert plan.tensor_canonical_data_type == DataTypesEnum.ts
    assert plan.input_compatible_data_type == DataTypesEnum.image


def test_build_industrial_task_plan_uses_detection_window_for_predict_horizon():
    plan = build_industrial_task_plan(
        'classification', {'data_type': 'time_series', 'detection_window': 7})

    assert plan.have_predict_horizon is True
    assert plan.task.task_type is TaskTypesEnum.ts_forecasting
    assert plan.task.task_params.forecast_length == 7


def test_build_industrial_task_plan_builds_regular_classification_task():
    plan = build_industrial_task_plan(
        'classification', {'data_type': 'tensor'})

    assert plan.have_predict_horizon is False
    assert plan.task.task_type is TaskTypesEnum.classification


def test_resolve_learning_strategy_flags_detects_big_and_tabular_contexts():
    flags = resolve_learning_strategy_flags(
        {'learning_strategy': 'ts2tabular_big'})

    assert flags.strategy_name == 'ts2tabular_big'
    assert flags.is_big_data is True
    assert flags.is_default_fedot_context is True


def test_should_use_predict_horizon_requires_time_series_and_window():
    assert should_use_predict_horizon(
        {'data_type': 'time_series', 'detection_window': 5}) is True
    assert should_use_predict_horizon({'data_type': 'tensor'}) is False
    assert should_use_predict_horizon(None) is False


def test_resolve_industrial_data_type_plan_rejects_unknown_type():
    with pytest.raises(ValueError, match='Unsupported industrial data_type'):
        resolve_industrial_data_type_plan({'data_type': 'unknown'})
