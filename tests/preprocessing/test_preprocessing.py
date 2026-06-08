from types import SimpleNamespace

import numpy as np
import pytest

import fedot.preprocessing.preprocessing as preprocessing_module
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_prepare_tensordata_uses_obligatory_bridge_path(monkeypatch):
    captured = {}
    preprocessor = DataPreprocessor()

    monkeypatch.setattr(preprocessing_module.Backend(), 'name', 'gpu')

    def fake_tensordata_to_input_data(tensor_data):
        captured['tensor_data'] = tensor_data
        return 'input-data'

    def fake_input_data_to_tensordata(input_data, backend_name, state):
        captured['converted_input'] = input_data
        captured['backend_name'] = backend_name
        captured['state'] = state
        return 'tensor-output'

    monkeypatch.setattr(
        preprocessing_module, 'tensordata_to_input_data', fake_tensordata_to_input_data)
    monkeypatch.setattr(
        preprocessing_module, 'input_data_to_tensordata', fake_input_data_to_tensordata)
    monkeypatch.setattr(
        preprocessor, 'obligatory_prepare_for_fit', lambda data: 'processed-input')

    tensor_data = SimpleNamespace(
        task=Task(TaskTypesEnum.classification), features=np.array([[1]]))
    result = preprocessor.prepare_tensordata(
        tensor_data, is_fit_stage=True, is_optional=False)

    assert result == 'tensor-output'
    assert captured['tensor_data'] is tensor_data
    assert captured['converted_input'] == 'processed-input'
    assert captured['backend_name'] == 'gpu'
    assert captured['state'].value == 'fit'


@pytest.mark.unit
def test_prepare_tensordata_uses_optional_predict_bridge_path(monkeypatch):
    captured = {}
    preprocessor = DataPreprocessor()

    monkeypatch.setattr(preprocessing_module.Backend(), 'name', 'cpu')

    def fake_tensordata_to_input_data(tensor_data):
        return 'input-data'

    def fake_input_data_to_tensordata(input_data, backend_name, state):
        captured['backend_name'] = backend_name
        captured['state'] = state
        captured['input_data'] = input_data
        return 'tensor-output'

    monkeypatch.setattr(
        preprocessing_module, 'tensordata_to_input_data', fake_tensordata_to_input_data)
    monkeypatch.setattr(
        preprocessing_module, 'input_data_to_tensordata', fake_input_data_to_tensordata)

    def fake_optional_prepare_for_predict(pipeline, data):
        captured['pipeline'] = pipeline
        captured['optional_input'] = data
        return 'processed-predict-input'

    monkeypatch.setattr(
        preprocessor, 'optional_prepare_for_predict', fake_optional_prepare_for_predict)

    pipeline = object()
    result = preprocessor.prepare_tensordata(
        SimpleNamespace(task=Task(TaskTypesEnum.classification),
                        features=np.array([[1]])),
        is_fit_stage=False,
        is_optional=True,
        pipeline=pipeline,
    )

    assert result == 'tensor-output'
    assert captured['pipeline'] is pipeline
    assert captured['optional_input'] == 'input-data'
    assert captured['input_data'] == 'processed-predict-input'
    assert captured['backend_name'] == 'cpu'
    assert captured['state'].value == 'predict'
