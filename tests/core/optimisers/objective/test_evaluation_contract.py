"""Observable evaluator behavior, including failures after a successful fold."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from golem.core.optimisers.objective import Objective

from fedot import create_data
from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.evaluation_contracts import (
    EvaluationComplete, EvaluationIncomplete, EvaluationReused, FailureKind,
    RetryPolicy, RetryableEvaluationError, ValidationResult,
)
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def data(offset=0):
    return TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table,
                      features=torch.arange(12, dtype=torch.float32).reshape(6, 2) + offset,
                      target=torch.arange(6, dtype=torch.float32), idx=np.arange(6))


class RecordingPipeline:
    """Controlled effect boundary: attempts really allocate and release model state."""
    descriptive_id = 'recording-pipeline'
    nodes = ('node',)

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls, self.cleanups, self.keys = 0, 0, []
        self.model = None

    @property
    def is_fitted(self):
        return self.model is not None

    def fit(self, tensor_data, **options):
        self.calls += 1
        self.keys.append(options['fold_id'])
        self.model = object()
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error

    def unfit(self):
        self.cleanups += 1
        self.model = None


def evaluator(folds=2, objective=None, **options):
    return PipelineObjectiveEvaluateWithTensorData(
        objective or Objective(lambda graph, **kwargs: float(graph.calls)),
        lambda: [(data(i), data(i + 10)) for i in range(folds)], **options)


def test_second_fold_failure_never_returns_partial_fitness():
    graph = RecordingPipeline([None, ValueError('second fold failed')])
    evaluate = evaluator()
    assert not evaluate.evaluate(graph).valid
    outcome = evaluate.last_outcome
    assert isinstance(outcome, EvaluationIncomplete)
    assert outcome.expected_folds == 2
    assert [fold.fold_id for fold in outcome.folds] == [0]
    assert outcome.failure.kind is FailureKind.FIT
    assert graph.cleanups == graph.calls == 2
    assert graph.model is None


@pytest.mark.parametrize('failure', [TimeoutError('deadline'), ValueError('invalid'), RuntimeError('backend')])
def test_nonretryable_error_runs_once_and_cleans(failure):
    graph = RecordingPipeline([failure])
    evaluate = evaluator(retry_policy=RetryPolicy(max_attempts=3))
    assert not evaluate.evaluate(graph).valid
    assert graph.calls == graph.cleanups == 1


def test_transient_failure_retries_only_current_fold():
    graph = RecordingPipeline([None, RetryableEvaluationError('temporary'), None])
    evaluate = evaluator(retry_policy=RetryPolicy(max_attempts=2))
    outcome = evaluate.evaluate_result(graph)
    assert isinstance(outcome, EvaluationComplete)
    assert [fold.attempts for fold in outcome.folds] == [1, 2]
    assert outcome.metrics == (2.0,)
    assert graph.calls == graph.cleanups == 3
    assert graph.keys[0] != graph.keys[1] == graph.keys[2]
    assert outcome.attempts[1].failure.kind is FailureKind.TRANSIENT


def test_retry_exhaustion_is_terminal_and_does_not_evaluate_next_fold():
    graph = RecordingPipeline([RetryableEvaluationError('temporary')] * 3)
    outcome = evaluator(3, retry_policy=RetryPolicy(2)).evaluate_result(graph)
    assert isinstance(outcome, EvaluationIncomplete)
    assert not outcome.folds
    assert graph.calls == graph.cleanups == 2


def test_timeout_is_retryable_only_when_explicitly_enabled():
    graph = RecordingPipeline([TimeoutError('deadline'), None])
    outcome = evaluator(1, retry_policy=RetryPolicy(2, (FailureKind.TIMEOUT,))).evaluate_result(graph)
    assert isinstance(outcome, EvaluationComplete)
    assert graph.cleanups == graph.calls == 2


def test_duplicate_candidate_reuses_complete_record_without_success_accounting():
    evaluate = evaluator()
    first, duplicate = RecordingPipeline(), RecordingPipeline()
    outcome = evaluate.evaluate_result(first)
    reused = evaluate.evaluate_result(duplicate)
    assert isinstance(reused, EvaluationReused)
    assert reused.original is outcome
    assert duplicate.calls == duplicate.cleanups == 0
    assert evaluate.evaluate(duplicate).valid
    with pytest.raises(FrozenInstanceError):
        outcome.candidate_id = 'other'
    evaluate.clear_results()
    assert isinstance(evaluate.evaluate_result(duplicate), EvaluationComplete)


def test_changed_dataset_does_not_reuse_previous_candidate():
    folds = [(data(), data(10))]
    evaluate = PipelineObjectiveEvaluateWithTensorData(Objective(lambda *a, **kw: 1.), lambda: folds)
    assert isinstance(evaluate.evaluate_result(RecordingPipeline()), EvaluationComplete)
    folds[0][0].features[4, 0] = 900
    graph = RecordingPipeline()
    assert isinstance(evaluate.evaluate_result(graph), EvaluationComplete)
    assert graph.calls == 1


def test_bounded_outcome_cache_evicts_oldest_candidate():
    evaluate = evaluator(1, result_cache_size=1)
    first, second = RecordingPipeline(), RecordingPipeline()
    second.descriptive_id = 'second'
    evaluate.evaluate_result(first)
    evaluate.evaluate_result(second)
    assert isinstance(evaluate.evaluate_result(first), EvaluationComplete)
    assert first.calls == 2


def test_duplicate_prefitted_graph_is_released_without_another_fit():
    evaluate = evaluator()
    evaluate.evaluate_result(RecordingPipeline())
    duplicate = RecordingPipeline()
    duplicate.model = object()
    assert isinstance(evaluate.evaluate_result(duplicate), EvaluationReused)
    assert duplicate.calls == 0 and duplicate.cleanups == 1
    assert not duplicate.is_fitted


def test_prefitted_graph_cannot_leak_training_state_across_folds():
    graph = RecordingPipeline()
    graph.model = object()
    outcome = evaluator().evaluate_result(graph)
    assert outcome.failure.kind is FailureKind.VALIDATION
    assert graph.calls == 0 and graph.cleanups == 1


@pytest.mark.parametrize('folds,expected', [(0, None), (1, 2), (3, 2)])
def test_fold_count_contract_rejects_empty_or_truncated_producer(folds, expected):
    graph = RecordingPipeline()
    outcome = evaluator(folds, expected_folds=expected).evaluate_result(graph)
    assert isinstance(outcome, EvaluationIncomplete)
    assert outcome.failure.kind is FailureKind.DATA
    assert graph.calls == 0


def test_generator_failure_is_not_treated_as_end_of_dataset():
    def producer():
        yield data(), data(1)
        raise ValueError('source read failed')
    evaluate = PipelineObjectiveEvaluateWithTensorData(Objective(lambda *a, **kw: 1.), producer)
    graph = RecordingPipeline()
    assert isinstance(evaluate.evaluate_result(graph), EvaluationIncomplete)
    assert graph.calls == 0


def test_validation_hook_stops_before_training():
    graph = RecordingPipeline()
    outcome = evaluator(validator=lambda _: ValidationResult(('unsupported operation',))).evaluate_result(graph)
    assert outcome.failure.kind is FailureKind.VALIDATION
    assert graph.calls == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_metric_is_invalid_and_releases_model(value):
    graph = RecordingPipeline()
    evaluate = evaluator(1, objective=Objective(lambda *a, **kw: value))
    assert not evaluate.evaluate(graph).valid
    assert evaluate.last_outcome.failure.kind is FailureKind.METRIC
    assert graph.cleanups == 1


def test_metric_error_is_preserved_by_strict_metrics_objective():
    calls = []

    def metric(graph, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RetryableEvaluationError('metric service temporarily unavailable')
        return 0.5

    graph = RecordingPipeline()
    outcome = evaluator(1, objective=MetricsObjective([metric]), retry_policy=RetryPolicy(2)).evaluate_result(graph)
    assert isinstance(outcome, EvaluationComplete)
    assert graph.calls == graph.cleanups == 2
    assert outcome.attempts[0].failure.kind is FailureKind.TRANSIENT


def test_unexpected_exception_propagates_after_cleanup():
    graph = RecordingPipeline([KeyError('programming error')])
    with pytest.raises(KeyError, match='programming error'):
        evaluator().evaluate(graph)
    assert graph.cleanups == 1


def test_cleanup_error_is_terminal_even_after_retryable_failure():
    graph = RecordingPipeline([RetryableEvaluationError('temporary')])
    graph.unfit = Mock(side_effect=OSError('cannot release'))
    outcome = evaluator(retry_policy=RetryPolicy(3)).evaluate_result(graph)
    assert outcome.failure.kind is FailureKind.CLEANUP
    assert outcome.attempts[0].failure.kind is FailureKind.TRANSIENT
    assert outcome.attempts[0].cleanup_failure.kind is FailureKind.CLEANUP
    assert graph.calls == graph.unfit.call_count == 1


def test_legacy_retention_keeps_only_final_success():
    graph = RecordingPipeline()
    outcome = evaluator(do_unfit=False).evaluate_result(graph)
    assert isinstance(outcome, EvaluationComplete)
    assert graph.cleanups == 1 and graph.model is not None
    graph.unfit()


def test_failed_attempt_discards_pending_prediction_writes():
    backend = Mock()
    backend.load_node_prediction.return_value = None
    sessions = []

    class WritingPipeline(RecordingPipeline):
        def fit(self, tensor_data, **options):
            session = options['predictions_cache']
            sessions.append(session)
            session.save_node_prediction('node', 'raw', options['fold_id'], tensor_data)
            super().fit(tensor_data, **options)

    graph = WritingPipeline([RetryableEvaluationError('temporary'), None])
    outcome = evaluator(1, retry_policy=RetryPolicy(2), predictions_cache=backend).evaluate_result(graph)
    assert isinstance(outcome, EvaluationComplete)
    assert backend.save_node_prediction.call_count == 1
    assert len(sessions) == 2
    assert all(session.cache is None and not session.pending for session in sessions)
    assert graph.cleanups == 2


def test_cleanup_failure_does_not_publish_prediction_cache():
    backend = Mock()

    class WritingPipeline(RecordingPipeline):
        def fit(self, tensor_data, **options):
            options['predictions_cache'].save_node_prediction('node', 'raw', options['fold_id'], tensor_data)
            super().fit(tensor_data, **options)

    graph = WritingPipeline()
    graph.unfit = Mock(side_effect=RuntimeError('cleanup failed'))
    outcome = evaluator(1, predictions_cache=backend).evaluate_result(graph)
    assert outcome.failure.kind is FailureKind.CLEANUP
    backend.save_node_prediction.assert_not_called()


def test_metric_dimension_failure_releases_model_with_legacy_retention():
    from golem.core.optimisers.objective.objective import to_fitness

    class VariableObjective:
        is_multi_objective = True
        metric_names = ('metric',)

        def __call__(self, graph, **kwargs):
            return to_fitness((1.,) * graph.calls, True)

    graph = RecordingPipeline()
    outcome = evaluator(objective=VariableObjective(), do_unfit=False).evaluate_result(graph)
    assert outcome.failure.kind is FailureKind.METRIC
    assert graph.cleanups == graph.calls == 2
    assert not graph.is_fitted


def test_real_operations_cache_is_scoped_to_full_fold_data(tmp_path):
    from fedot.core.caching.operations_cache import OperationsCache

    class CachedPipeline:
        descriptive_id = 'cached-model'

        def __init__(self):
            self.nodes = [SimpleNamespace(descriptive_id='cached-node', fitted_operation=None)]
            self.model_builds = 0

        @property
        def is_fitted(self):
            return self.nodes[0].fitted_operation is not None

        def try_load_from_cache(self, cache, fold_id):
            cache.try_load_nodes(self.nodes, fold_id)

        def fit(self, tensor_data, **options):
            if not self.is_fitted:
                self.nodes[0].fitted_operation = {'mean': float(tensor_data.features.mean())}
                self.model_builds += 1

        def unfit(self):
            self.nodes[0].fitted_operation = None

    cache_dir = tmp_path / 'operations'
    cache_dir.mkdir()
    cache = OperationsCache(cache_dir=str(cache_dir))
    folds = [(data(), data(10))]
    evaluate = PipelineObjectiveEvaluateWithTensorData(
        Objective(lambda graph, **kwargs: graph.nodes[0].fitted_operation['mean']), lambda: folds,
        operations_cache=cache, result_cache_size=0)
    first, second, changed = CachedPipeline(), CachedPipeline(), CachedPipeline()
    original = evaluate.evaluate_result(first)
    same = evaluate.evaluate_result(second)
    folds[0][0].features[4, 0] += 100
    different = evaluate.evaluate_result(changed)
    assert all(isinstance(outcome, EvaluationComplete) for outcome in (original, same, different))
    assert first.model_builds == changed.model_builds == 1
    assert second.model_builds == 0
    assert original.metrics == same.metrics != different.metrics
    assert not any(graph.is_fitted for graph in (first, second, changed))


@pytest.mark.integration
def test_real_cpu_pipeline_complete_evaluation_does_not_mutate_data():
    train = create_data(np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.]], dtype=np.float32),
                        target=np.array([0, 1, 0, 1]), use_cache=False)
    before = deepcopy(train)
    node = PrimaryNode('torch_linear')
    node.parameters = {'epochs': 2, 'learning_rate': 0.01}
    graph = Pipeline(node)

    def error_rate(model, reference_data, **kwargs):
        predicted = model.predict(reference_data).predict
        return float(np.mean(np.asarray(predicted) != np.asarray(reference_data.target)))

    evaluate = PipelineObjectiveEvaluateWithTensorData(
        MetricsObjective([error_rate]), lambda: [(train, deepcopy(train)), (deepcopy(train), train)], expected_folds=2)
    result = evaluate.evaluate_result(graph)
    assert isinstance(result, EvaluationComplete), result
    assert len(result.folds) == 2
    assert not graph.is_fitted
    assert train == before
    assert train.preparation_state == before.preparation_state
