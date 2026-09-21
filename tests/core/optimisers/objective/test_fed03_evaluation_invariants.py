"""Independent invariants for the FED-03 evaluation contract."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from golem.core.optimisers.objective import Objective
from hypothesis import given, strategies as st

from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.evaluation_contracts import (
    EvaluationComplete,
    EvaluationIncomplete,
    FailureKind,
    RetryPolicy,
    RetryableEvaluationError,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def tensor_data(offset=0):
    return TensorData(
        Task(TaskTypesEnum.regression),
        DataTypesEnum.table,
        features=torch.arange(12, dtype=torch.float32).reshape(6, 2) + offset,
        target=torch.arange(6, dtype=torch.float32),
        idx=np.arange(6),
    )


class AttemptPipeline:
    nodes = ('node',)
    descriptive_id = 'fed03-invariant-candidate'

    def __init__(self, events=(), publish_prediction=False):
        self.events = list(events)
        self.publish_prediction = publish_prediction
        self.calls = 0
        self.cleanups = 0
        self.model = None

    @property
    def is_fitted(self):
        return self.model is not None

    def fit(self, data, **options):
        self.calls += 1
        self.model = object()
        if self.publish_prediction:
            options['predictions_cache'].save_node_prediction(
                'node', 'raw', options['fold_id'], deepcopy(data))
        if self.events:
            event = self.events.pop(0)
            if event is not None:
                raise event

    def unfit(self):
        self.cleanups += 1
        self.model = None


def evaluator(folds, pipeline_objective=None, **options):
    def source():
        return [(tensor_data(i), tensor_data(i + 100)) for i in range(folds)]

    objective = pipeline_objective or Objective(
        lambda graph, **kwargs: float(graph.calls))
    return PipelineObjectiveEvaluateWithTensorData(objective, source, expected_folds=folds, **options)


@given(
    fold_count=st.integers(min_value=2, max_value=8),
    failed_fold=st.integers(min_value=1, max_value=7),
)
def test_any_failed_fold_rejects_the_whole_evaluation(fold_count, failed_fold):
    """A successful prefix must never be exposed as aggregate fitness."""
    failed_fold %= fold_count
    events = [None] * failed_fold + [ValueError('fold failed')]
    graph = AttemptPipeline(events)

    result = evaluator(fold_count).evaluate_result(graph)

    assert isinstance(result, EvaluationIncomplete)
    assert result.expected_folds == fold_count
    assert tuple(record.fold_id for record in result.folds) == tuple(
        range(failed_fold))
    assert result.failure.kind is FailureKind.FIT
    assert graph.calls == graph.cleanups == failed_fold + 1
    assert graph.model is None


@given(
    transient_failures=st.integers(min_value=0, max_value=6),
    max_attempts=st.integers(min_value=1, max_value=6),
)
def test_retry_budget_is_deterministic_and_cleanup_owned_by_each_attempt(transient_failures, max_attempts):
    events = [RetryableEvaluationError(
        'temporary')] * transient_failures + [None]
    graph = AttemptPipeline(events)
    result = evaluator(1, retry_policy=RetryPolicy(
        max_attempts)).evaluate_result(graph)
    expected_attempts = min(transient_failures + 1, max_attempts)

    assert graph.calls == graph.cleanups == expected_attempts
    assert len(result.attempts) == expected_attempts
    assert graph.model is None
    if transient_failures < max_attempts:
        assert isinstance(result, EvaluationComplete)
        assert result.folds[0].attempts == transient_failures + 1
    else:
        assert isinstance(result, EvaluationIncomplete)
        assert result.failure.kind is FailureKind.TRANSIENT


@given(retryable_prefix=st.integers(min_value=0, max_value=5))
def test_permanent_failure_stops_after_retryable_prefix(retryable_prefix):
    graph = AttemptPipeline(
        [RetryableEvaluationError('temporary')] * retryable_prefix + [ValueError('permanent')])
    result = evaluator(2, retry_policy=RetryPolicy(
        max_attempts=retryable_prefix + 3)).evaluate_result(graph)

    assert isinstance(result, EvaluationIncomplete)
    assert result.failure.kind is FailureKind.FIT
    assert graph.calls == graph.cleanups == retryable_prefix + 1
    assert not result.folds


class PredictionBackend:
    def __init__(self):
        self.published = []

    def save_node_prediction(self, *args):
        self.published.append(args)

    def load_node_prediction(self, *args):
        return None


def test_timeout_discards_attempt_local_predictions_and_fitted_state():
    backend = PredictionBackend()
    graph = AttemptPipeline([TimeoutError('deadline')],
                            publish_prediction=True)

    result = evaluator(1, predictions_cache=backend).evaluate_result(graph)

    assert isinstance(result, EvaluationIncomplete)
    assert result.failure.kind is FailureKind.TIMEOUT
    assert backend.published == []
    assert graph.calls == graph.cleanups == 1
    assert graph.model is None


def test_unexpected_exception_discards_attempt_local_predictions_and_fitted_state():
    backend = PredictionBackend()
    graph = AttemptPipeline([KeyError('bug')], publish_prediction=True)

    with pytest.raises(KeyError, match='bug'):
        evaluator(1, predictions_cache=backend).evaluate_result(graph)

    assert backend.published == []
    assert graph.calls == graph.cleanups == 1
    assert graph.model is None
