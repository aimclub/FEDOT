from types import SimpleNamespace

import numpy as np
from pymonad.either import Left, Right

import fedot.api.api_utils.assumptions.assumptions_handler as handler_module
from fedot.api.api_utils.assumptions.assumptions_handler import AssumptionsHandler
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum


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
    assert isinstance(result.monoid[0].exception, RuntimeError)


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


def test_fit_assumption_and_check_correctness_raises_from_original_exception(monkeypatch):
    handler = AssumptionsHandler(SimpleNamespace())
    pipeline = _FakePipeline()
    original_error = RuntimeError('fit failed')
    fit_error = SimpleNamespace(message='boom', exception=original_error)
    monkeypatch.setattr(handler, 'try_fit_assumption', lambda **kwargs: Left(fit_error))

    try:
        handler.fit_assumption_and_check_correctness(pipeline)
    except ValueError as error:
        assert str(error) == 'boom'
        assert error.__cause__ is original_error
    else:
        raise AssertionError('Compatibility wrapper should chain original fit error')


def test_propose_assumptions_preserves_explicit_gpu_bridge_operations():
    input_data = InputData.from_numpy(
        features_array=np.array([
            [0.1, 1.0],
            [0.3, 0.5],
            [0.2, 0.8],
            [0.9, 0.4],
        ]),
        target_array=np.array([0, 1, 0, 1]),
        task='classification',
        data_type=DataTypesEnum.table,
    )

    assumptions = AssumptionsHandler(input_data).propose_assumptions(
        initial_assumption=None,
        available_operations=['industrial_inception_nn', 'industrial_resnet_nn'],
    )

    pipeline = assumptions[0]
    operation_types = {node.operation.operation_type for node in pipeline.nodes}

    assert pipeline.root_node.operation.operation_type in {'industrial_inception_nn', 'industrial_resnet_nn'}
    assert 'rf' not in operation_types
    assert 'rfr' not in operation_types