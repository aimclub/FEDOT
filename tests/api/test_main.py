import numpy as np
import pytest

from fedot import Fedot
from fedot.core.data.input_data.data import OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.common.enums import StateEnum
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class _StubPipeline(Pipeline):
    def __init__(self):
        super().__init__()
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


def _minimal_tensordata_for_predict() -> TensorData:
    """Minimal TensorData in predict state for facade tests (bridge + pipeline receive real TensorData)."""
    return TensorData(
        state=StateEnum.PREDICT,
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=np.array([0, 1]),
        target=None,
    )


def _minimal_tensordata_for_fit() -> TensorData:
    """Minimal TensorData in fit state for predefined fit / tune facade tests."""
    return TensorData(
        state=StateEnum.FIT,
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=np.array([0, 1]),
        target=np.array([0, 1]),
    )


def _minimal_tensordata_ts_predict() -> TensorData:
    """Minimal TensorData for TS forecast-style facade tests."""
    return TensorData(
        state=StateEnum.PREDICT,
        features=np.array([[1.0], [2.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        data_type=DataTypesEnum.ts,
        idx=np.array([0, 1]),
        target=None,
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

    prediction = model.predict_tensordata(tensor_data=_minimal_tensordata_for_predict(), output_mode='labels')

    assert prediction.shape == (2, 2)
    assert model.current_pipeline.calls == [('predict_tensordata', 'labels')]


def test_main_facade_predict_proba_tensordata_uses_service_rule_mode_selection():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()

    prediction = model.predict_proba_tensordata(tensor_data=_minimal_tensordata_for_predict(),
                                                probs_for_all_classes=True)

    assert prediction.shape == (2, 2)
    assert model.current_pipeline.calls == [('predict_tensordata', 'full_probs')]


def test_main_facade_fit_tensordata_uses_predefined_runtime_path(monkeypatch):
    model = Fedot(problem='classification')
    tensor_data = _minimal_tensordata_for_fit()

    class FakePredefinedModel:
        def __init__(self, predefined_model, data, log, use_input_preprocessing=True, api_preprocessor=None):
            assert predefined_model == 'logit'
            assert data is tensor_data

        def fit_tensordata(self):
            return _StubPipeline()

    monkeypatch.setattr('fedot.api.main.PredefinedModel', FakePredefinedModel)

    pipeline = model.fit_tensordata(tensor_data=tensor_data, predefined_model='logit')

    assert isinstance(pipeline, _StubPipeline)


def test_main_facade_fit_tensordata_rejects_composition_path():
    model = Fedot(problem='classification')

    with pytest.raises(ValueError, match='supports only predefined models or pipelines'):
        model.fit_tensordata(tensor_data=_minimal_tensordata_for_fit(), predefined_model=None)


def test_main_facade_fit_tensordata_stores_legacy_train_data(monkeypatch):
    model = Fedot(problem='classification')
    tensor_data = _minimal_tensordata_for_fit()
    stored_train_data = type('StoredTrainData', (), {'target': 'stored-target'})()
    model.data_processor.to_input_data = lambda td: stored_train_data

    class FakePredefinedModel:
        def __init__(self, predefined_model, data, log, use_input_preprocessing=True, api_preprocessor=None):
            pass

        def fit_tensordata(self):
            return _StubPipeline()

    monkeypatch.setattr('fedot.api.main.PredefinedModel', FakePredefinedModel)
    monkeypatch.setattr('fedot.api.main.graph_structure', lambda pipeline: 'pipeline-structure')

    pipeline = model.fit_tensordata(tensor_data=tensor_data, predefined_model='logit')

    assert isinstance(pipeline, _StubPipeline)
    assert model.train_data is stored_train_data
    assert model.target == 'stored-target'


def test_main_facade_predict_tensordata_stores_legacy_test_data():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    stored_test_data = object()
    model.data_processor.to_input_data = lambda td: stored_test_data

    model.predict_tensordata(tensor_data=_minimal_tensordata_for_predict(), output_mode='labels')

    assert model.test_data is stored_test_data


def test_main_facade_tune_tensordata_uses_tensor_tuner_runtime_path(monkeypatch):
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    model.train_data = type('TrainData', (), {'target': 'train-target'})()
    converted_input = type('ConvertedInput', (), {'target': 'converted-target'})()
    model.data_processor.to_input_data = lambda tensor_data: converted_input
    merged_preprocessor = object()
    captured = {}

    class FakeTunedPipeline(_StubPipeline):
        def __init__(self):
            super().__init__()
            self.preprocessor = 'pipeline-preprocessor'

        def fit_tensordata(self, tensor_data):
            captured['refit_tensor_data'] = tensor_data

    class FakeTuner:
        was_tuned = True

        def tune(self, pipeline, show_progress=False):
            captured['tune_pipeline'] = pipeline
            captured['show_progress'] = show_progress
            return FakeTunedPipeline()

    class FakeBuilder:
        def __init__(self, task):
            captured['task'] = task

        def with_tuner(self, tuner):
            captured['tuner_class'] = tuner
            return self

        def with_cv_folds(self, cv_folds):
            captured['cv_folds'] = cv_folds
            return self

        def with_n_jobs(self, n_jobs):
            captured['n_jobs'] = n_jobs
            return self

        def with_metric(self, metric):
            captured['metric'] = metric
            return self

        def with_iterations(self, iterations):
            captured['iterations'] = iterations
            return self

        def with_timeout(self, timeout):
            captured['timeout'] = timeout
            return self

        def build_tensordata(self, tensor_data):
            captured['tensor_data'] = tensor_data
            return FakeTuner()

    def fake_merge_preprocessors(api_preprocessor, pipeline_preprocessor, use_auto_preprocessing):
        captured['api_preprocessor'] = api_preprocessor
        captured['pipeline_preprocessor'] = pipeline_preprocessor
        captured['use_auto_preprocessing'] = use_auto_preprocessing
        return merged_preprocessor

    monkeypatch.setattr('fedot.api.main.TunerBuilder', FakeBuilder)
    monkeypatch.setattr('fedot.api.main.BasePreprocessor.merge_preprocessors', fake_merge_preprocessors)

    tensor_data = _minimal_tensordata_for_fit()
    result = model.tune_tensordata(
        tensor_data=tensor_data,
        metric_name='roc_auc',
        iterations=5,
        timeout=2.5,
        cv_folds=3,
        n_jobs=4,
        show_progress=True,
    )

    assert isinstance(result, FakeTunedPipeline)
    assert captured['tensor_data'] is tensor_data
    assert captured['cv_folds'] == 3
    assert captured['n_jobs'] == 4
    assert captured['metric'] == 'roc_auc'
    assert captured['iterations'] == 5
    assert captured['timeout'] == 2.5
    assert captured['show_progress'] is True
    assert captured['tune_pipeline'].__class__ is _StubPipeline
    assert captured['refit_tensor_data'] is tensor_data
    assert captured['api_preprocessor'] is model.data_processor.preprocessor
    assert captured['pipeline_preprocessor'] == 'pipeline-preprocessor'
    assert captured['use_auto_preprocessing'] == model.params.get('use_auto_preprocessing')
    assert model.api_composer.was_tuned is True
    assert model.train_data is converted_input
    assert model.target == 'converted-target'
    assert model.current_pipeline.preprocessor is merged_preprocessor


def test_main_facade_get_metrics_tensordata_uses_tensor_prediction_and_legacy_metrics_flow():
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    stored_test_data = object()
    model.data_processor.to_input_data = lambda tensor_data: stored_test_data
    captured = {}

    def fake_get_metrics(**kwargs):
        captured.update(kwargs)
        return {'roc_auc': 0.9}

    model.get_metrics = fake_get_metrics

    result = model.get_metrics_tensordata(
        tensor_data=_minimal_tensordata_for_predict(),
        target=np.array([0, 1]),
        metric_names=['roc_auc'],
        rounding_order=4,
    )

    assert result == {'roc_auc': 0.9}
    assert model.test_data is stored_test_data
    assert model.current_pipeline.calls == [('predict_tensordata', 'default')]
    assert np.array_equal(captured['target'], np.array([0, 1]))
    assert captured['metric_names'] == ['roc_auc']
    assert captured['rounding_order'] == 4


def test_main_facade_explain_tensordata_uses_legacy_conversion_boundary(monkeypatch):
    model = Fedot(problem='classification')
    model.current_pipeline = _StubPipeline()
    stored_test_data = object()
    model.data_processor.to_input_data = lambda tensor_data: stored_test_data
    captured = {}

    def fake_explain_pipeline(pipeline, data, method, visualization, **kwargs):
        captured['pipeline'] = pipeline
        captured['data'] = data
        captured['method'] = method
        captured['visualization'] = visualization
        captured['kwargs'] = kwargs
        return 'explainer'

    monkeypatch.setattr('fedot.api.main.explain_pipeline', fake_explain_pipeline)

    result = model.explain_tensordata(
        tensor_data=_minimal_tensordata_for_predict(),
        method='surrogate_dt',
        visualization=False,
        max_depth=3,
    )

    assert result == 'explainer'
    assert captured['pipeline'] is model.current_pipeline
    assert captured['data'] is stored_test_data
    assert captured['method'] == 'surrogate_dt'
    assert captured['visualization'] is False
    assert captured['kwargs'] == {'max_depth': 3}


def test_main_facade_forecast_tensordata_uses_legacy_conversion_boundary(monkeypatch):
    model = Fedot(problem='ts_forecasting', task_params=TsForecastingParams(forecast_length=2))
    model.current_pipeline = object()
    model.train_data = type('TrainData', (), {
        'task': Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2))
    })()
    stored_test_data = type('TestData', (), {'target': np.array([1.0, 2.0])})()
    model.data_processor.to_input_data = lambda tensor_data: stored_test_data
    captured = {}

    def fake_out_of_sample_ts_forecast(pipeline, test_data, horizon):
        captured['pipeline'] = pipeline
        captured['test_data'] = test_data
        captured['horizon'] = horizon
        return np.array([10.0, 11.0, 12.0])

    def fake_convert_forecast_to_output(test_data, predict):
        captured['converted_test_data'] = test_data
        captured['predict'] = predict
        return OutputData(
            idx=np.arange(len(predict)),
            predict=predict,
            target=None,
            task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
            data_type=DataTypesEnum.ts,
        )

    monkeypatch.setattr('fedot.api.main.out_of_sample_ts_forecast', fake_out_of_sample_ts_forecast)
    monkeypatch.setattr('fedot.api.main.convert_forecast_to_output', fake_convert_forecast_to_output)

    result = model.forecast_tensordata(tensor_data=_minimal_tensordata_ts_predict(), horizon=3)

    assert np.array_equal(result, np.array([10.0, 11.0, 12.0]))
    assert model.test_data is stored_test_data
    assert model.test_data.target is None
    assert captured['pipeline'] is model.current_pipeline
    assert captured['test_data'] is stored_test_data
    assert captured['converted_test_data'] is stored_test_data
    assert captured['horizon'] == 3
    assert np.array_equal(captured['predict'], np.array([10.0, 11.0, 12.0]))


def test_main_facade_forecast_tensordata_reuses_forecast_validation():
    model = Fedot(problem='classification')
    model.current_pipeline = object()

    with pytest.raises(ValueError, match='Forecasting can be used only for the time series'):
        model.forecast_tensordata(tensor_data=_minimal_tensordata_ts_predict())


def test_main_facade_fit_tensordata_merges_api_and_pipeline_preprocessors(monkeypatch):
    model = Fedot(problem='classification')
    stored_train_data = type('StoredTrainData', (), {'target': 'stored-target'})()
    model.data_processor.to_input_data = lambda tensor_data: stored_train_data
    merged_preprocessor = object()
    captured = {}

    class FakePipeline(_StubPipeline):
        def __init__(self):
            super().__init__()
            self.preprocessor = 'pipeline-preprocessor'

    class FakePredefinedModel:
        def __init__(self, predefined_model, data, log, use_input_preprocessing=True, api_preprocessor=None):
            pass

        def fit_tensordata(self):
            return FakePipeline()

    def fake_merge_preprocessors(api_preprocessor, pipeline_preprocessor, use_auto_preprocessing):
        captured['api_preprocessor'] = api_preprocessor
        captured['pipeline_preprocessor'] = pipeline_preprocessor
        captured['use_auto_preprocessing'] = use_auto_preprocessing
        return merged_preprocessor

    monkeypatch.setattr('fedot.api.main.PredefinedModel', FakePredefinedModel)
    monkeypatch.setattr('fedot.api.main.BasePreprocessor.merge_preprocessors', fake_merge_preprocessors)
    monkeypatch.setattr('fedot.api.main.graph_structure', lambda pipeline: 'pipeline-structure')

    pipeline = model.fit_tensordata(tensor_data=_minimal_tensordata_for_fit(), predefined_model='logit')

    assert isinstance(pipeline, FakePipeline)
    assert captured['api_preprocessor'] is model.data_processor.preprocessor
    assert captured['pipeline_preprocessor'] == 'pipeline-preprocessor'
    assert captured['use_auto_preprocessing'] == model.params.get('use_auto_preprocessing')
    assert model.current_pipeline.preprocessor is merged_preprocessor
