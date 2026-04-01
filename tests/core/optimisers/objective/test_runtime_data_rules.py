import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.optimisers.objective.runtime_data_rules import (
    build_benchmark_runtime_plan,
    build_runtime_fit_plan,
    build_runtime_predict_plan,
    normalize_benchmark_runtime_mode,
    with_input_runtime,
    with_tensor_runtime,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_input_data():
    return InputData(
        idx=np.arange(4),
        features=np.arange(8).reshape(4, 2),
        target=np.array([0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_runtime_rules_normalize_supported_modes():
    assert normalize_benchmark_runtime_mode(' tensor_gpu_bridge ') == 'tensor_gpu_bridge'

    with pytest.raises(ValueError, match='Unsupported benchmark runtime mode'):
        normalize_benchmark_runtime_mode('unknown_mode')


def test_runtime_rules_build_runtime_plan_for_tensor_gpu_bridge():
    plan = build_benchmark_runtime_plan('tensor_gpu_bridge')

    assert plan.mode == 'tensor_gpu_bridge'
    assert plan.use_tensor_train is True
    assert plan.use_gpu_bridge is True
    assert plan.tensor_backend_name == 'gpu'
    assert plan.operation_device == 'gpu'


def test_runtime_rules_build_fit_and_predict_plans_for_input_and_tensor_modes():
    input_data = _make_input_data()
    tensor_runtime_data = with_tensor_runtime(
        input_data=input_data,
        tensor_data='tensor-data',
        runtime_mode='tensor_cpu',
    )
    input_runtime_data = with_input_runtime(
        input_data=input_data,
        runtime_mode='input_gpu_bridge',
    )

    tensor_fit_plan = build_runtime_fit_plan(tensor_runtime_data)
    tensor_predict_plan = build_runtime_predict_plan(tensor_runtime_data)
    input_fit_plan = build_runtime_fit_plan(input_runtime_data)
    input_predict_plan = build_runtime_predict_plan(input_runtime_data)

    assert tensor_fit_plan.fit_method_name == 'fit_tensordata'
    assert tensor_fit_plan.fit_data == 'tensor-data'
    assert tensor_predict_plan.predict_method_name == 'predict_tensordata'
    assert tensor_predict_plan.predict_data == 'tensor-data'

    assert input_fit_plan.fit_method_name == 'fit'
    assert input_fit_plan.fit_data is input_data
    assert input_predict_plan.predict_method_name == 'predict'
    assert input_predict_plan.predict_data is input_data

