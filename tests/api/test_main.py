import numpy as np
import pytest

from fedot import Fedot
from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _StubPipeline:
    def __init__(self):
        self.calls = []

    def predict(self, test_data, output_mode='default'):
        self.calls.append(('predict', output_mode))
        return OutputData(
            idx=np.arange(2),
            predict=np.array([[0.2, 0.8], [0.7, 0.3]]),
            target=None,
            task=Task(TaskTypesEnum.classification),
            data_type=DataTypesEnum.table,
        )

    def predict_tensordata(self, tensor_data, output_mode='default'):
        self.calls.append(('predict_tensordata', output_mode))
        return OutputData(
            idx=np.arange(2),
            predict=np.array([[0.2, 0.8], [0.7, 0.3]]),
            target=None,
            task=Task(TaskTypesEnum.classification),
            data_type=DataTypesEnum.table,
        )


def test_main_facade_raises_not_fitted_errors_for_predictive_methods():
    model = Fedot(problem='classification')

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.predict(features=np.array([[1.0]]))

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.tune()

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.get_metrics()

    with pytest.raises(ValueError, match='Model not fitted yet'):
        model.return_report()


def test_main_facade_predict_proba_rejects_non_classification_tasks():
    model = Fedot(problem='regression')
    model.current_pipeline = object()

    with pytest.raises(ValueError, match='Probabilities of predictions are available only for classification'):
        model.predict_proba(features=np.array([[1.0]]))


def test_main_facade_uses_service_rule_for_predict_proba_mode_selection():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    model.target = 'target'
    model.data_processor.define_data = lambda **kwargs: type('Input',
                                                             (), {'task': Task(TaskTypesEnum.classification)})()

    model.predict_proba(features=np.array([[1.0], [2.0]]), probs_for_all_classes=True)

    assert model.current_pipeline.calls == [('predict', 'full_probs')]


def test_main_facade_forecast_requires_time_series_task():
    model = Fedot(problem='classification')
    model.current_pipeline = object()

    with pytest.raises(ValueError, match='Forecasting can be used only for the time series'):
        model.forecast()


def test_main_facade_predict_tensordata_uses_tensor_pipeline_entrypoint():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()

    prediction = model.predict_tensordata(tensor_data=object(), output_mode='labels')

    assert prediction.shape == (2, 2)
    assert model.current_pipeline.calls == [('predict_tensordata', 'labels')]


def test_main_facade_predict_proba_tensordata_uses_service_rule_mode_selection():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()

    prediction = model.predict_proba_tensordata(tensor_data=object(), probs_for_all_classes=True)

    assert prediction.shape == (2, 2)
    assert model.current_pipeline.calls == [('predict_tensordata', 'full_probs')]


def test_main_facade_fit_tensordata_uses_predefined_runtime_path(monkeypatch):
    model = Fedot(problem='classification')

    class FakePredefinedModel:
        def __init__(self, predefined_model, data, log, use_input_preprocessing=True, api_preprocessor=None):
            assert predefined_model == 'logit'
            assert data == 'tensor-data'

        def fit_tensordata(self):
            return _StubPipeline()

    monkeypatch.setattr('fedot.api.main.PredefinedModel', FakePredefinedModel)

    pipeline = model.fit_tensordata(tensor_data='tensor-data', predefined_model='logit')

    assert isinstance(pipeline, _StubPipeline)


def test_main_facade_fit_tensordata_rejects_composition_path():
    model = Fedot(problem='classification')

    with pytest.raises(ValueError, match='supports only predefined models or pipelines'):
        model.fit_tensordata(tensor_data='tensor-data', predefined_model=None)
