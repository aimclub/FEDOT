"""Effectful evaluator; decisions and outcomes live in evaluation_contracts."""
import sys
from collections import OrderedDict
from dataclasses import replace
from datetime import timedelta
from math import isfinite
from typing import Callable, Iterable, Optional, Tuple

from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective.objective import Objective, to_fitness
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate

from fedot.core.caching.evaluation_context import TensorDataCacheContext
from fedot.core.caching.evaluation_session import PredictionCacheSession, snapshot_operations
from fedot.core.caching.normalization import stable_hash
from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.data.merge.data_merger import DataMergeError
from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.evaluation_contracts import (
    AttemptRecord, EvaluationComplete, EvaluationFailure, EvaluationIncomplete,
    EvaluationOutcome, EvaluationReused, FailureKind, FoldRecord,
    PipelineValidator, RetryPolicy, failure_from_exception, should_retry, validate_pipeline,
)
from fedot.core.pipelines.pipeline import Pipeline

TensorDataSource = Callable[[], Iterable[Tuple[TensorData, TensorData]]]
DataSource = TensorDataSource
EXPECTED_ERRORS = (TimeoutError, DataMergeError, ValueError, TypeError, RuntimeError, ArithmeticError)


class PipelineObjectiveEvaluateWithTensorData(ObjectiveEvaluate[Pipeline]):
    """Evaluate every fold or return invalid fitness, never a partial average.

    ``evaluate`` keeps the GOLEM Fitness API. ``evaluate_result`` exposes the
    immutable diagnostic outcome. Retries are opt-in and scoped to one fold.
    ``do_unfit=False`` retains only the final successful model for legacy callers;
    intermediate and failed attempts always release their fitted state.
    """

    def __init__(self, objective: Objective, data_producer: TensorDataSource,
                 time_constraint: Optional[timedelta] = None,
                 validation_blocks: Optional[int] = None,
                 operations_cache: Optional[OperationsCache] = None,
                 predictions_cache: Optional[PredictionsCache] = None,
                 eval_n_jobs: int = 1, do_unfit: bool = True, *,
                 retry_policy: RetryPolicy = RetryPolicy(),
                 validator: PipelineValidator = validate_pipeline,
                 expected_folds: Optional[int] = None,
                 cache_namespace: str = 'fedot-evaluation-v1',
                 result_cache_size: int = 256):
        super().__init__(objective, eval_n_jobs=eval_n_jobs)
        if expected_folds is not None and (
                isinstance(expected_folds, bool) or not isinstance(expected_folds, int) or expected_folds < 1):
            raise ValueError('expected_folds must be a positive integer or None')
        if isinstance(result_cache_size, bool) or not isinstance(result_cache_size, int) or result_cache_size < 0:
            raise ValueError('result_cache_size must be a nonnegative integer')
        if not isinstance(retry_policy, RetryPolicy):
            raise TypeError('retry_policy must be RetryPolicy')
        if not isinstance(cache_namespace, str) or not cache_namespace.strip():
            raise ValueError('cache_namespace must be a nonempty string')
        self._data_producer = data_producer
        self._time_constraint = time_constraint
        self._validation_blocks = validation_blocks
        self._operations_cache = operations_cache
        self._predictions_cache = predictions_cache
        self._log = default_log(self)
        self._do_unfit = do_unfit
        self.retry_policy = retry_policy
        self.validator = validator
        self.expected_folds = expected_folds
        self.cache_namespace = cache_namespace
        self.result_cache_size = result_cache_size
        self._completed = OrderedDict()
        self._fold_context_version = None
        self._fold_context_templates = ()
        self.last_outcome: Optional[EvaluationOutcome] = None

    def evaluate(self, graph: Pipeline) -> Fitness:
        outcome = self.evaluate_result(graph)
        if isinstance(outcome, EvaluationReused):
            outcome = outcome.original
        values = outcome.metrics if isinstance(outcome, EvaluationComplete) else None
        return to_fitness(values, self._objective.is_multi_objective)

    def evaluate_result(self, graph: Pipeline) -> EvaluationOutcome:
        """Own the lifecycle of a single candidate. No TensorData is retained."""
        graph.log = self._log
        candidate_id = graph.descriptive_id
        self.last_outcome = None
        try:
            validation = self.validator(graph)
            if not validation.valid:
                return self._reject(graph, FailureKind.VALIDATION, '; '.join(validation.violations))
            folds = tuple(self._data_producer())
            count = self.expected_folds if self.expected_folds is not None else len(folds)
            if not folds or len(folds) != count:
                return self._reject(graph, FailureKind.DATA,
                                    f'expected {count} folds, received {len(folds)}', count)
            contexts = self._build_fold_contexts(folds, candidate_id)
        except EXPECTED_ERRORS as error:
            return self._reject(graph, FailureKind.DATA, str(error))

        evaluation_key = stable_hash((tuple(c.key for c in contexts), self._validation_blocks,
                                      tuple(self._objective.metric_names), self._eval_n_jobs,
                                      str(self._time_constraint), self._objective.is_multi_objective))
        if self._do_unfit and evaluation_key in self._completed:
            self._completed.move_to_end(evaluation_key)
            if graph.is_fitted:
                cleanup = self._release(graph)
                if cleanup is not None:
                    self.last_outcome = EvaluationIncomplete(candidate_id, count, (), (), cleanup)
                    return self.last_outcome
            self.last_outcome = EvaluationReused(self._completed[evaluation_key])
            return self.last_outcome
        if graph.is_fitted and len(folds) > 1:
            return self._reject(graph, FailureKind.VALIDATION,
                                'a prefitted pipeline cannot be reused across multiple folds', count)

        records, attempts = [], []
        for fold_id, ((train, test), context) in enumerate(zip(folds, contexts)):
            for attempt in range(1, self.retry_policy.max_attempts + 1):
                metrics, failure, cleanup_failure = self._attempt(
                    graph, train, test, context.key, fold_id == count - 1,
                    len(records[0].metrics) if records else None)
                attempts.append(AttemptRecord(fold_id, attempt, failure, cleanup_failure))
                if cleanup_failure is not None:
                    failure = cleanup_failure
                if failure is None:
                    records.append(FoldRecord(fold_id, metrics, attempt, context.key))
                    break
                if not should_retry(self.retry_policy, failure, attempt):
                    self.last_outcome = EvaluationIncomplete(
                        candidate_id, count, tuple(records), tuple(attempts), failure)
                    return self.last_outcome

        outcome = EvaluationComplete(candidate_id, count, tuple(records), tuple(attempts))
        if self._do_unfit and self.result_cache_size:
            self._completed[evaluation_key] = outcome
            while len(self._completed) > self.result_cache_size:
                self._completed.popitem(last=False)
        self.last_outcome = outcome
        return outcome

    def _attempt(self, graph, train, test, cache_key, final_fold, expected_metrics):
        metrics, failure, cleanup_failure = (), None, None
        phase = FailureKind.FIT
        session = PredictionCacheSession(self._predictions_cache) if self._predictions_cache is not None else None
        operations = []
        try:
            prepared = self.prepare_graph(graph, train, cache_key, self._eval_n_jobs, prediction_cache=session)
            phase = FailureKind.METRIC
            objective = getattr(self._objective, 'evaluate_strict', self._objective)
            fitness = objective(prepared, reference_data=test,
                                validation_blocks=self._validation_blocks,
                                predictions_cache=session, fold_id=cache_key)
            metrics = tuple(float(x) for x in fitness.values) if fitness.valid else ()
            if not metrics or not all(map(isfinite, metrics)):
                failure = EvaluationFailure(FailureKind.METRIC, 'objective returned invalid or nonfinite fitness')
            elif expected_metrics is not None and len(metrics) != expected_metrics:
                failure = EvaluationFailure(FailureKind.METRIC, 'metric dimension changed between folds')
            if failure is None:
                if self._operations_cache is not None:
                    operations = snapshot_operations(graph)
        except EXPECTED_ERRORS as error:
            failure = failure_from_exception(error, phase)
        finally:
            # Unexpected exceptions also release state, then propagate unchanged.
            retain = not self._do_unfit and final_fold and failure is None and metrics and sys.exc_info()[0] is None
            if not retain:
                cleanup_failure = self._release(graph)
            try:
                if failure is None and cleanup_failure is None and sys.exc_info()[0] is None:
                    if self._operations_cache is not None:
                        self._operations_cache.save_nodes(operations, cache_key)
                    if session is not None:
                        session.commit()
            except (OSError, ValueError, RuntimeError) as error:
                failure = EvaluationFailure(FailureKind.CACHE, str(error), type(error).__name__)
                if retain:
                    cleanup_failure = self._release(graph)
            finally:
                operations.clear()
                if session is not None:
                    session.close()
        return metrics, failure, cleanup_failure

    @staticmethod
    def _release(graph):
        try:
            graph.unfit()
        except Exception as error:
            # Resource-release errors are terminal and never retryable.
            return EvaluationFailure(FailureKind.CLEANUP, str(error), type(error).__name__)
        return None

    def _reject(self, graph, kind, message, count=0):
        failure = EvaluationFailure(kind, message)
        if graph.is_fitted:
            failure = self._release(graph) or failure
        self.last_outcome = EvaluationIncomplete(
            graph.descriptive_id, count, (), (), failure)
        return self.last_outcome

    def clear_results(self):
        """Release bounded diagnostic memoization at the end of a session."""
        self._completed.clear()
        self._fold_context_version = None
        self._fold_context_templates = ()
        self.last_outcome = None

    def _build_fold_contexts(self, folds, candidate_id):
        """Reuse data identities only for an explicitly versioned split source."""
        data_version = getattr(self._data_producer, 'evaluation_data_version', None)
        if data_version is None:
            return tuple(TensorDataCacheContext.from_fold(
                train, test, fold_id, candidate_id, self.cache_namespace)
                for fold_id, (train, test) in enumerate(folds))

        from fedot.extensions.registry import registered_extensions_identity

        context_version = (data_version, registered_extensions_identity())
        if context_version != self._fold_context_version:
            self._fold_context_templates = tuple(TensorDataCacheContext.from_fold(
                train, test, fold_id, '__evaluation_data__', self.cache_namespace)
                for fold_id, (train, test) in enumerate(folds))
            self._fold_context_version = context_version
        return tuple(replace(context, candidate_id=candidate_id)
                     for context in self._fold_context_templates)

    def prepare_graph(self, graph: Pipeline, train_data: TensorData,
                      fold_id=None, n_jobs: int = -1, *, prediction_cache=None) -> Pipeline:
        if graph.is_fitted:
            return graph
        if self._operations_cache is not None:
            graph.try_load_from_cache(self._operations_cache, fold_id)
        graph.fit(train_data, n_jobs=n_jobs, time_constraint=self._time_constraint,
                  predictions_cache=self._predictions_cache if prediction_cache is None else prediction_cache,
                  fold_id=fold_id)
        return graph

    def evaluate_intermediate_metrics(self, graph: Pipeline):
        """Intermediate-node metrics are not part of the TensorData contract yet."""

    @property
    def tensor_data(self):
        return self._data_producer.args[0]
