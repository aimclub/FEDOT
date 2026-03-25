from types import SimpleNamespace

import numpy as np

import fedot.api.api_utils.api_data as api_data_module
from fedot.api.api_utils.api_data import ApiDataProcessor
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
