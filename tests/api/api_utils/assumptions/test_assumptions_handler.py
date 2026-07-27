from types import SimpleNamespace

import pytest
from pymonad.either import Left, Right

import fedot.api.api_utils.assumptions.assumptions_handler as handler_module
from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler


class _FakePipeline:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.loaded = False
        self.fitted = False
        self.predicted = False

    def try_load_from_cache(self, operations_cache, preprocessing_cache):
        self.loaded = True

    def fit(self, data_train, n_jobs=-1):
        if self.should_fail:
            raise RuntimeError('fit failed')
        self.fitted = True
        self.fit_data = data_train
        self.fit_n_jobs = n_jobs

    def predict(self, data_test):
        self.predicted = True
        self.predict_data = data_test
        return 'ok'


def test_fit_assumption_and_check_correctness_keeps_compatibility_wrapper(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    monkeypatch.setattr(handler, 'try_fit_assumption',
                        lambda **kwargs: Right(pipeline))

    assert handler.fit_assumption_and_check_correctness(pipeline) is pipeline

    monkeypatch.setattr(handler, 'try_fit_assumption',
                        lambda **kwargs: Left(SimpleNamespace(message='boom')))

    try:
        handler.fit_assumption_and_check_correctness(pipeline)
    except ValueError as error:
        assert str(error) == 'boom'
    else:
        raise AssertionError(
            'Compatibility wrapper should raise ValueError for failed assumption fitting')


def test_fit_assumption_and_check_correctness_raises_from_original_exception(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    original_error = RuntimeError('fit failed')
    fit_error = SimpleNamespace(message='boom', exception=original_error)
    monkeypatch.setattr(handler, 'try_fit_assumption',
                        lambda **kwargs: Left(fit_error))

    try:
        handler.fit_assumption_and_check_correctness(pipeline)
    except ValueError as error:
        assert str(error) == 'boom'
        assert error.__cause__ is original_error
    else:
        raise AssertionError(
            'Compatibility wrapper should chain original fit error')


def test_propose_assumptions_with_tensordata_builds_auto_assumption(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    captured = {}

    class _FakeAssumptionsBuilder:
        @classmethod
        def get(cls, data):
            captured['data'] = data
            return cls()

        def from_operations(self, available_operations):
            captured['available_operations'] = available_operations
            return self

        def build(self, use_input_preprocessing=True):
            captured['use_input_preprocessing'] = use_input_preprocessing
            return [pipeline]

    monkeypatch.setattr(handler_module, 'AssumptionsBuilder', _FakeAssumptionsBuilder)

    result = handler.propose_assumptions(None, available_operations=['torch_linear'])

    assert result == [pipeline]
    assert captured['data'] is handler.data
    assert captured['available_operations'] == ['torch_linear']
    assert captured['use_input_preprocessing'] is False


def test_propose_assumptions_with_tensordata_accepts_user_pipeline_list():
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()

    assert handler.propose_assumptions([pipeline]) == [pipeline]


def test_try_fit_assumption_returns_right_for_success(monkeypatch):
    class _FakeTensorDataSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, data):
            assert data.name == 'tensor-data'

            def _producer():
                yield 'tensor-train', 'tensor-test'

            return _producer

    monkeypatch.setattr(handler_module, 'DataSourceSplitter', _FakeTensorDataSplitter)
    monkeypatch.setattr(handler_module.MemoryAnalytics, 'log',
                        staticmethod(lambda *args, **kwargs: None))

    pipeline = _FakePipeline()
    result = AssumptionsHandler(SimpleNamespace(name='tensor-data')).try_fit_assumption(
        pipeline,
        eval_n_jobs=3,
    )

    assert result.is_right()
    assert result.value is pipeline
    assert pipeline.loaded is True
    assert pipeline.fitted is True
    assert pipeline.predicted is True
    assert pipeline.fit_data == 'tensor-train'
    assert pipeline.predict_data == 'tensor-test'
    assert pipeline.fit_n_jobs == 3


def test_fit_assumption_and_check_correctness_raises_on_failure(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    monkeypatch.setattr(handler, 'try_fit_assumption',
                        lambda **kwargs: Left(SimpleNamespace(message='tensor boom')))

    with pytest.raises(ValueError, match='tensor boom'):
        handler.fit_assumption_and_check_correctness(pipeline)
