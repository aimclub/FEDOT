from types import SimpleNamespace

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

    def predict(self, data_test):
        self.predicted = True
        return 'ok'


def test_try_fit_assumption_returns_right_for_success(monkeypatch):
    monkeypatch.setattr(handler_module, 'train_test_data_setup', lambda data: ('train', 'test'))
    monkeypatch.setattr(handler_module.MemoryAnalytics, 'log', staticmethod(lambda *args, **kwargs: None))

    pipeline = _FakePipeline()
    result = AssumptionsHandler(SimpleNamespace()).try_fit_assumption(pipeline)

    assert result.is_right()
    assert result.value is pipeline
    assert pipeline.loaded is True
    assert pipeline.fitted is True
    assert pipeline.predicted is True


def test_try_fit_assumption_returns_left_for_expected_fit_failure(monkeypatch):
    monkeypatch.setattr(handler_module, 'train_test_data_setup', lambda data: ('train', 'test'))

    pipeline = _FakePipeline(should_fail=True)
    result = AssumptionsHandler(SimpleNamespace()).try_fit_assumption(pipeline)

    assert result.is_left()
    assert result.monoid[0].code == 'initial_assumption_fit_failed'
    assert 'fit failed' in result.monoid[0].message


def test_fit_assumption_and_check_correctness_keeps_compatibility_wrapper(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    monkeypatch.setattr(handler, 'try_fit_assumption', lambda **kwargs: Right(pipeline))

    assert handler.fit_assumption_and_check_correctness(pipeline) is pipeline

    monkeypatch.setattr(handler, 'try_fit_assumption', lambda **kwargs: Left(SimpleNamespace(message='boom')))

    try:
        handler.fit_assumption_and_check_correctness(pipeline)
    except ValueError as error:
        assert str(error) == 'boom'
    else:
        raise AssertionError('Compatibility wrapper should raise ValueError for failed assumption fitting')
