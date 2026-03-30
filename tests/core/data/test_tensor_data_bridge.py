from types import SimpleNamespace

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.tensor_data_bridge import tensordata_to_input_data
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_tensordata_to_input_data_returns_legacy_inputdata_shape():
    features = np.array([[1, 2], [3, 4]])
    target = np.array([0, 1])
    tensor_data = SimpleNamespace(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        state='fit',
        idx=np.array([5, 6]),
        features=features,
        target=target,
        features_names=np.array(['a', 'b']),
        categorical_idx=np.array([1]),
    )

    result = tensordata_to_input_data(tensor_data)

    assert isinstance(result, InputData)
    assert result.data_type == DataTypesEnum.table
    assert np.array_equal(result.idx, np.array([5, 6]))
    assert np.array_equal(result.features, features)
    assert np.array_equal(result.target, target)
    assert np.array_equal(result.features_names, np.array(['a', 'b']))
    assert np.array_equal(result.categorical_idx, np.array([1]))
    assert result.features is not features
    assert result.target is not target


@pytest.mark.unit
def test_tensordata_to_input_data_keeps_predict_target_empty():
    tensor_data = SimpleNamespace(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        state='predict',
        idx=None,
        features=np.array([[10], [20]]),
        target=np.array([1, 0]),
        features_names=None,
        categorical_idx=None,
    )

    result = tensordata_to_input_data(tensor_data)

    assert result.target is None
    assert np.array_equal(result.idx, np.array([0, 1]))
