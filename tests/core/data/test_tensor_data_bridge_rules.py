from types import SimpleNamespace

import numpy as np
import pytest

from fedot.core.data.tensor_data_bridge_rules import (
    build_tensordata_input_bridge_plan,
    normalize_input_bridge_target,
    resolve_input_idx,
)
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_build_tensordata_input_bridge_plan_maps_to_input_compatible_type():
    tensor_data = SimpleNamespace(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        state=StateEnum.FIT,
        idx=np.array([10, 11]),
        features=np.array([[1, 2], [3, 4]]),
        target=np.array([0, 1]),
        features_names=np.array(['f1', 'f2']),
        categorical_idx=np.array([1]),
    )

    plan = build_tensordata_input_bridge_plan(tensor_data)

    assert plan.data_type == DataTypesEnum.table
    assert np.array_equal(plan.idx, np.array([10, 11]))
    assert np.array_equal(plan.target, np.array([0, 1]))
    assert np.array_equal(plan.features_names, np.array(['f1', 'f2']))
    assert np.array_equal(plan.categorical_idx, np.array([1]))


@pytest.mark.unit
def test_build_tensordata_input_bridge_plan_drops_target_for_predict_state():
    tensor_data = SimpleNamespace(
        task=Task(TaskTypesEnum.ts_forecasting),
        data_type=DataTypesEnum.ts,
        state='predict',
        idx=None,
        features=np.array([[1.0], [2.0], [3.0]]),
        target=np.array([1.0, 2.0, 3.0]),
        features_names=None,
        categorical_idx=None,
    )

    plan = build_tensordata_input_bridge_plan(tensor_data)

    assert plan.state == StateEnum.PREDICT
    assert plan.data_type == DataTypesEnum.ts
    assert plan.target is None
    assert np.array_equal(plan.idx, np.array([0, 1, 2]))


@pytest.mark.unit
def test_resolve_input_idx_infers_range_from_sample_axis_not_feature_axis():
    features = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    result = resolve_input_idx(idx=None, features=features)

    assert np.array_equal(result, np.array([0, 1]))


@pytest.mark.unit
def test_normalize_input_bridge_target_returns_copy():
    target = np.array([1, 2, 3])

    normalized = normalize_input_bridge_target(target, StateEnum.FIT)

    assert np.array_equal(normalized, target)
    assert normalized is not target

