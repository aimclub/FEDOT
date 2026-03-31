from types import SimpleNamespace

import numpy as np

import fedot.api.api_utils.api_data as api_data_module
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.tools import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class _PassthroughPreprocessor:
    def obligatory_prepare_for_fit(self, data):
        return data

    def obligatory_prepare_for_predict(self, data):
        return data

    def optional_prepare_for_fit(self, pipeline, data):
        return data

    def optional_prepare_for_predict(self, pipeline, data):
        return data

    def convert_indexes_for_fit(self, pipeline, data):
        return data

    def convert_indexes_for_predict(self, pipeline, data):
        return data

    def update_indices_for_time_series(self, data):
        return data

    def reduce_memory_size(self, data):
        return data


def test_define_data_does_not_mutate_original_multimodal_features(monkeypatch):
    original_features = {'idx': np.array([10, 11]), 'table': np.array([[1], [2]])}

    def fake_strategy_selector(features, target=None, task=None, is_predict=None):
        assert 'idx' not in features
        return {'table': SimpleNamespace(idx=None)}

    monkeypatch.setattr(api_data_module, 'data_strategy_selector', fake_strategy_selector)

    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    processor.preprocessor = _PassthroughPreprocessor()

    result = processor.define_data(features=original_features, target=np.array([0, 1]), is_predict=False)

    assert 'idx' in original_features
    assert np.array_equal(original_features['idx'], np.array([10, 11]))
    assert np.array_equal(result['table'].idx, np.array([10, 11]))


def test_define_predictions_uses_in_sample_forecasting_plan(monkeypatch):
    captured = {}

    def fake_in_sample_ts_forecast(pipeline, test_data, horizon):
        captured['horizon'] = horizon
        return np.array([1.0] * horizon)

    def fake_convert_forecast_to_output(test_data, forecast, idx):
        captured['idx'] = idx
        return SimpleNamespace(predict=forecast, idx=idx)

    monkeypatch.setattr(api_data_module, 'in_sample_ts_forecast', fake_in_sample_ts_forecast)
    monkeypatch.setattr(api_data_module, 'convert_forecast_to_output', fake_convert_forecast_to_output)

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2))
    processor = ApiDataProcessor(task, use_input_preprocessing=False)
    test_data = SimpleNamespace(task=task, idx=np.array([0, 1, 2, 3, 4]))

    prediction = processor.define_predictions(current_pipeline=object(), test_data=test_data,
                                              in_sample=True, validation_blocks=2)

    assert captured['horizon'] == 4
    assert np.array_equal(captured['idx'], np.array([1, 2, 3, 4]))
    assert len(prediction.predict) == 4


def test_to_tensordata_uses_bridge_adapter_for_fit(monkeypatch):
    captured = {}

    def fake_input_data_to_tensordata(input_data, backend_name, state):
        captured['input_data'] = input_data
        captured['backend_name'] = backend_name
        captured['state'] = state
        return 'tensor-data'

    monkeypatch.setattr(api_data_module, 'input_data_to_tensordata', fake_input_data_to_tensordata)

    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    result = processor.to_tensordata(input_data, backend_name='gpu', is_predict=False)

    assert result == 'tensor-data'
    assert captured['input_data'] is input_data
    assert captured['backend_name'] == 'gpu'
    assert captured['state'] == StateEnum.FIT


def test_to_tensordata_uses_predict_state_for_inference(monkeypatch):
    captured = {}

    def fake_input_data_to_tensordata(input_data, backend_name, state):
        captured['state'] = state
        return 'tensor-data'

    monkeypatch.setattr(api_data_module, 'input_data_to_tensordata', fake_input_data_to_tensordata)

    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    input_data = InputData(
        idx=np.array([0]),
        features=np.array([[1]]),
        target=None,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    processor.to_tensordata(input_data, backend_name='cpu', is_predict=True)

    assert captured['state'] == StateEnum.PREDICT


def test_to_input_data_uses_reverse_bridge_adapter(monkeypatch):
    captured = {}

    def fake_tensordata_to_input_data(tensor_data):
        captured['tensor_data'] = tensor_data
        return 'input-data'

    monkeypatch.setattr(api_data_module, 'tensordata_to_input_data', fake_tensordata_to_input_data)

    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    tensor_data = SimpleNamespace(features=np.array([[1]]))

    result = processor.to_input_data(tensor_data)

    assert result == 'input-data'
    assert captured['tensor_data'] is tensor_data


def test_to_input_data_returns_legacy_inputdata_from_tensor_boundary(monkeypatch):
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    def fake_tensordata_to_input_data(tensor_data):
        return input_data

    monkeypatch.setattr(api_data_module, 'tensordata_to_input_data', fake_tensordata_to_input_data)

    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)

    result = processor.to_input_data(SimpleNamespace(features=np.array([[1]])))

    assert result is input_data


def test_define_tensordata_uses_explicit_input_and_tensor_boundaries(monkeypatch):
    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )
    captured = {}

    processor.define_data = lambda features, target=None, is_predict=False: input_data

    def fake_to_tensordata(input_data_arg, backend_name='cpu', is_predict=False):
        captured['input_data'] = input_data_arg
        captured['backend_name'] = backend_name
        captured['is_predict'] = is_predict
        return 'tensor-data'

    processor.to_tensordata = fake_to_tensordata

    result = processor.define_tensordata(
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        is_predict=True,
        backend_name='gpu',
    )

    assert result == 'tensor-data'
    assert captured['input_data'] is input_data
    assert captured['backend_name'] == 'gpu'
    assert captured['is_predict'] is True


def test_define_tensordata_rejects_multimodal_legacy_path():
    processor = ApiDataProcessor(Task(TaskTypesEnum.classification), use_input_preprocessing=False)
    input_data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )
    multimodal = MultiModalData({'main': input_data})
    processor.define_data = lambda features, target=None, is_predict=False: multimodal

    with pytest.raises(ValueError, match='supports only InputData'):
        processor.define_tensordata(features={'main': np.array([[1], [2]])})
