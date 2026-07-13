from types import SimpleNamespace

from fedot.api.api_utils.assumptions.assumptions_handler_rules import (
    build_assumption_fit_error,
    decide_preset,
    normalize_initial_assumption,
    resolve_initial_assumption,
)
from fedot.api.api_utils.schemas import raise_from_assumption_fit_error
from fedot.core.pipelines.pipeline import Pipeline


class _FakePipeline(Pipeline):
    pass


def test_normalize_initial_assumption_handles_none_single_and_list():
    pipeline = _FakePipeline()

    assert normalize_initial_assumption(None) is None
    assert normalize_initial_assumption(pipeline) == [pipeline]
    assert normalize_initial_assumption([pipeline]) == [pipeline]


def test_resolve_initial_assumption_uses_builder_only_when_needed():
    pipeline = _FakePipeline()
    calls = {'count': 0}

    def builder():
        calls['count'] += 1
        return [pipeline]

    assert resolve_initial_assumption(None, builder) == [pipeline]
    assert calls['count'] == 1
    assert resolve_initial_assumption(pipeline, builder) == [pipeline]
    assert calls['count'] == 1


def test_build_assumption_fit_error_returns_typed_error_message():
    error = build_assumption_fit_error(RuntimeError('broken fit'))

    assert error.code == 'initial_assumption_fit_failed'
    assert 'broken fit' in error.message
    assert error.cause == 'broken fit'


def test_decide_preset_changes_only_for_auto_like_values():
    timer = SimpleNamespace()
    calls = {'count': 0}

    def chooser(_timer, n_jobs):
        calls['count'] += 1
        assert n_jobs == 2
        return 'fast_train'

    changed = decide_preset(None, timer, 2, chooser)
    unchanged = decide_preset('best_quality', timer, 2, chooser)

    assert changed.preset == 'fast_train'
    assert changed.was_changed is True
    assert unchanged.preset == 'best_quality'
    assert unchanged.was_changed is False
    assert calls['count'] == 1


def test_raise_from_assumption_fit_error_raises_value_error_with_cause():
    original_error = RuntimeError('fit failed')
    fit_error = build_assumption_fit_error(original_error)

    try:
        raise_from_assumption_fit_error(fit_error)
    except ValueError as error:
        assert 'fit failed' in str(error)
        assert error.__cause__ is original_error
    else:
        raise AssertionError('Expected ValueError from assumption fit error schema')
