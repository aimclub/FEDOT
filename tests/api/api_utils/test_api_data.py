from types import SimpleNamespace

import numpy as np
import torch

from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def test_define_predictions_delegates_to_pipeline_predict():
    # TODO @romankuklo: in-sample forecasting via ``in_sample_ts_forecast`` is currently
    # unwired in ``define_predictions`` (dead/commented-out code, unrelated to preprocessing
    # removal); it unconditionally delegates to ``current_pipeline.predict``.
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=2))
    processor = ApiDataProcessor(task)
    test_data = SimpleNamespace(task=task, idx=np.array([0, 1, 2, 3, 4]))

    captured = {}

    class StubPipeline:
        def predict(self, data):
            captured['data'] = data
            return SimpleNamespace(predict=np.array([1.0, 2.0]))

    prediction = processor.define_predictions(
        current_pipeline=StubPipeline(), test_data=test_data,
        in_sample=True, validation_blocks=2)

    assert captured['data'] is test_data
    assert np.array_equal(prediction.predict, np.array([1.0, 2.0]))


def test_define_predictions_flattens_out_of_sample_forecast():
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=2))
    processor = ApiDataProcessor(task)
    test_data = SimpleNamespace(task=task, idx=np.array([0, 1]))

    class StubPipeline:
        def predict(self, data):
            return SimpleNamespace(predict=torch.tensor([[0.0], [1.0]]))

    prediction = processor.define_predictions(
        current_pipeline=StubPipeline(), test_data=test_data, in_sample=False)

    assert torch.equal(prediction.predict, torch.tensor([0.0, 1.0]))


def test_accept_and_apply_recommendations_warns_and_does_not_mutate_input_data(caplog):
    # ``ApiDataProcessor`` no longer owns a ``DataPreprocessor``/``DummyPreprocessor``,
    # so recommendations from ``InputAnalyser`` are ignored with an explicit warning.
    processor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    features = np.array([['a', 1], ['b', 2]], dtype=object)
    data = SimpleNamespace(features=features)

    with caplog.at_level('WARNING'):
        result = processor.accept_and_apply_recommendations(
            data, {'label_encoded': {}})

    assert result is None
    assert data.features is features
    assert any(
        'Ignoring preprocessing recommendations' in record.message
        for record in caplog.records
    )


def test_accept_and_apply_recommendations_skips_warning_for_empty_recommendations(caplog):
    processor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    features = np.array([[1], [2]])
    data = SimpleNamespace(features=features)

    with caplog.at_level('WARNING'):
        processor.accept_and_apply_recommendations(data, {})

    assert data.features is features
    assert not any(
        'Ignoring preprocessing recommendations' in record.message
        for record in caplog.records
    )


def test_accept_and_apply_recommendations_warns_for_multimodal_data(caplog):
    processor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    inner = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1], [2]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )
    multimodal = MultiModalData({'source': inner})

    with caplog.at_level('WARNING'):
        processor.accept_and_apply_recommendations(
            multimodal, {'source': {}})

    assert multimodal['source'] is inner
    assert any(
        'Ignoring preprocessing recommendations' in record.message
        for record in caplog.records
    )
