import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.input_data_bridge import input_data_to_tensordata
from fedot.core.data.tensor_data_bridge import tensordata_to_input_data
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
    assert np.array_equal(captured['kwargs']['idx'], np.array([0, 1]))
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


@pytest.mark.unit
def test_input_data_tensor_round_trip_preserves_idx(monkeypatch):
    input_data = InputData(
        idx=np.array([10, 20]),
        features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        target=np.array([1, 0]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features_names=np.array(['f1', 'f2']),
        categorical_idx=np.array([1]),
    )

    def fake_create(source_data, backend_name, **kwargs):
        return type('TensorDataStub', (), {
            'task': kwargs['task'],
            'data_type': kwargs['data_type'],
            'state': kwargs['state'],
            'idx': np.array(kwargs['idx'], copy=True),
            'features': np.array(source_data, copy=True),
            'target': np.array(kwargs['target'], copy=True),
            'features_names': np.array(kwargs['features_names'], copy=True),
            'categorical_idx': np.array(kwargs['categorical_idx'], copy=True),
        })()

    monkeypatch.setattr('fedot.core.data.input_data_bridge.TensorData.create', fake_create)

    tensor_data = input_data_to_tensordata(input_data, backend_name='cpu', state=StateEnum.FIT)
    round_trip = tensordata_to_input_data(tensor_data)

    assert np.array_equal(round_trip.idx, input_data.idx)
    assert np.array_equal(round_trip.features, input_data.features)
    assert np.array_equal(round_trip.target, input_data.target)

