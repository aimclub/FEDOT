import numpy as np
import pytest

from fedot.core.data.input_data_bridge_rules import (
    build_input_data_tensor_bridge_plan,
    normalize_bridge_state,
)
from fedot.core.data.input_data_descriptor import InputDataDescriptor
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _descriptor(data_type: DataTypesEnum) -> InputDataDescriptor:
    return InputDataDescriptor(
        task=Task(TaskTypesEnum.classification),
        original_data_type=data_type,
        tensor_canonical_data_type=DataTypesEnum.tabular if data_type == DataTypesEnum.text else DataTypesEnum.ts,
        input_compatible_data_type=data_type,
        idx=np.array([0, 1]),
        features_names=np.array(['feature']),
        categorical_idx=np.array([0]),
        target_is_present=True,
    )


@pytest.mark.unit
def test_normalize_bridge_state_accepts_string_and_enum():
    assert normalize_bridge_state('fit') == StateEnum.FIT
    assert normalize_bridge_state(StateEnum.PREDICT) == StateEnum.PREDICT


@pytest.mark.unit
def test_build_input_data_tensor_bridge_plan_uses_tensor_canonical_type_and_copies_metadata():
    descriptor = _descriptor(DataTypesEnum.text)

    plan = build_input_data_tensor_bridge_plan(
        descriptor=descriptor,
        target=np.array([0, 1]),
        state='fit',
    )

    assert plan.data_type == DataTypesEnum.tabular
    assert plan.state == StateEnum.FIT
    assert np.array_equal(plan.target, np.array([0, 1]))
    assert np.array_equal(plan.features_names, np.array(['feature']))
    assert np.array_equal(plan.categorical_idx, np.array([0]))
    assert plan.features_names is not descriptor.features_names
    assert plan.categorical_idx is not descriptor.categorical_idx


@pytest.mark.unit
def test_build_input_data_tensor_bridge_plan_drops_target_for_predict_state():
    descriptor = _descriptor(DataTypesEnum.image)

    plan = build_input_data_tensor_bridge_plan(
        descriptor=descriptor,
        target=np.array([0, 1]),
        state=StateEnum.PREDICT,
    )

    assert plan.data_type == DataTypesEnum.ts
    assert plan.state == StateEnum.PREDICT
    assert plan.target is None
