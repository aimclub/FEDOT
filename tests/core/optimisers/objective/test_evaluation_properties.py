from hypothesis import given, strategies as st
import pytest

from fedot.core.optimisers.objective.evaluation_contracts import (
    EvaluationComplete, EvaluationFailure, FailureKind, FoldRecord, RetryPolicy, should_retry,
)


@given(st.integers(min_value=1, max_value=20), st.sampled_from(list(FailureKind)))
def test_retry_policy_has_finite_prefix_and_never_retries_permanent_errors(limit, kind):
    policy = RetryPolicy(limit, (FailureKind.TRANSIENT, FailureKind.TIMEOUT))
    failure = EvaluationFailure(kind, 'test')
    decisions = [should_retry(policy, failure, attempt) for attempt in range(1, limit + 2)]
    expected = [kind in (FailureKind.TRANSIENT, FailureKind.TIMEOUT)] * (limit - 1) + [False, False]
    assert decisions == expected


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=25))
def test_full_evaluation_averages_all_folds_and_rejects_any_missing_fold(values):
    records = tuple(FoldRecord(i, (value,), 1, str(i)) for i, value in enumerate(values))
    result = EvaluationComplete('candidate', len(records), records, ())
    assert result.metrics[0] == pytest.approx(sum(values) / len(values), abs=1e-8)
    for missing in range(len(records)):
        with pytest.raises(ValueError, match='every fold'):
            EvaluationComplete('candidate', len(records), records[:missing] + records[missing + 1:], ())


@pytest.mark.parametrize('kwargs', [dict(max_attempts=0), dict(max_attempts=True), dict(max_attempts=1.5),
                                    dict(retryable=('timeout',)), dict(retryable=(FailureKind.FIT,)),
                                    dict(retryable=(FailureKind.CLEANUP,))])
def test_retry_config_never_coerces_invalid_transport_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize('factory', [
    lambda: FoldRecord(True, (1.0,), 1, 'cache'),
    lambda: FoldRecord(0, (1.0,), True, 'cache'),
    lambda: EvaluationComplete('candidate', True, (FoldRecord(0, (1.0,), 1, 'cache'),), ()),
])
def test_evaluation_records_reject_boolean_integer_fields(factory):
    with pytest.raises(ValueError):
        factory()
