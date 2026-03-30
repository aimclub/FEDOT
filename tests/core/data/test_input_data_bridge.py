import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.input_data_bridge import input_data_to_tensordata
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_input_data_to_tensordata_passes_bridge_plan_to_tensor_create(monkeypatch):
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1, 2], [3, 4]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.text,
        features_names=np.array(['text', 'meta']),
        categorical_idx=np.array([0]),
    )

    captured = {}

    def fake_create(source_data, backend_name, **kwargs):
        captured['source_data'] = source_data
        captured['backend_name'] = backend_name
        captured['kwargs'] = kwargs
        return 'tensor-data'

    monkeypatch.setattr('fedot.core.data.input_data_bridge.TensorData.create', fake_create)

    result = input_data_to_tensordata(input_data, backend_name='cpu', state='fit')

    assert result == 'tensor-data'
    assert np.array_equal(captured['source_data'], input_data.features)
    assert captured['backend_name'] == 'cpu'
    assert captured['kwargs']['data_type'] == DataTypesEnum.tabular
    assert captured['kwargs']['state'] == StateEnum.FIT
    assert np.array_equal(captured['kwargs']['target'], np.array([0, 1]))
    assert np.array_equal(captured['kwargs']['features_names'], np.array(['text', 'meta']))
    assert np.array_equal(captured['kwargs']['categorical_idx'], np.array([0]))


@pytest.mark.unit
def test_input_data_to_tensordata_drops_target_for_predict_state(monkeypatch):
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1, 2], [3, 4]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.image,
    )

    captured = {}

    def fake_create(source_data, backend_name, **kwargs):
        captured['kwargs'] = kwargs
        return 'tensor-data'

    monkeypatch.setattr('fedot.core.data.input_data_bridge.TensorData.create', fake_create)

    input_data_to_tensordata(input_data, backend_name='gpu', state=StateEnum.PREDICT)

    assert captured['kwargs']['data_type'] == DataTypesEnum.ts
    assert captured['kwargs']['state'] == StateEnum.PREDICT
    assert captured['kwargs']['target'] is None
