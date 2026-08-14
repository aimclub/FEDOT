from collections import OrderedDict
from datetime import timedelta
from math import isfinite
from typing import Dict, Optional

from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective.objective import Objective, to_fitness
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate

from fedot.core.caching.evaluation_context import tensor_data_identity
from fedot.core.caching.normalization import stable_hash
from fedot.core.composer.metrics import ComplexityMetric, QualityMetric, StructuralComplexity
from fedot.core.data.bridges import tensordata_to_input_data
from fedot.core.data.merge.data_merger import DataMergeError
from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.evaluation_contracts import (
    AttemptRecord,
    EvaluationComplete,
    EvaluationFailure,
    EvaluationIncomplete,
    EvaluationOutcome,
    EvaluationReused,
    FailureKind,
    FoldRecord,
    PipelineValidator,
    RetryPolicy,
    failure_from_exception,
    should_retry,
    validate_pipeline,
)
from fedot.core.optimisers.objective.oof_objective_rules import build_oof_metric_data
from fedot.core.pipelines.oof import PipelineOOFExecutor
from fedot.core.pipelines.pipeline import Pipeline

EXPECTED_ERRORS = (TimeoutError, DataMergeError, ValueError, TypeError, RuntimeError, ArithmeticError)


class PipelineOOFObjectiveEvaluate(ObjectiveEvaluate[Pipeline]):
    """Score stitched out-of-fold predictions under the evaluation contract.

    TensorData is the public boundary. The recursive executor temporarily uses
    InputData because the classic pipeline operations still own that execution
    path. One complete record represents the atomic stitched OOF result; its
    internal split count is part of the cache identity.
    """

    def __init__(self,
                 objective: Objective,
                 tensor_data: TensorData,
                 cv_folds: int,
                 time_constraint: Optional[timedelta] = None,
                 eval_n_jobs: int = 1,
                 random_seed: int = 42,
                 do_unfit: bool = True,
                 *,
                 retry_policy: RetryPolicy = RetryPolicy(),
                 validator: PipelineValidator = validate_pipeline,
                 cache_namespace: str = 'fedot-evaluation-v1',
                 result_cache_size: int = 256):
        super().__init__(objective, eval_n_jobs=eval_n_jobs)
        if not isinstance(tensor_data, TensorData):
            raise TypeError('tensor_data must be TensorData')
        if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
            raise ValueError('cv_folds must be an integer greater than one')
        if isinstance(result_cache_size, bool) or not isinstance(result_cache_size, int) or result_cache_size < 0:
            raise ValueError('result_cache_size must be a nonnegative integer')
        if not isinstance(retry_policy, RetryPolicy):
            raise TypeError('retry_policy must be RetryPolicy')
        if not isinstance(cache_namespace, str) or not cache_namespace.strip():
            raise ValueError('cache_namespace must be a nonempty string')
        self._tensor_data = tensor_data
        self._cv_folds = cv_folds
        self._time_constraint = time_constraint
        self._random_seed = random_seed
        self._do_unfit = do_unfit
        self.retry_policy = retry_policy
        self.validator = validator
        self.cache_namespace = cache_namespace
        self.result_cache_size = result_cache_size
        self._completed = OrderedDict()
        self.last_outcome: Optional[EvaluationOutcome] = None
        self._log = default_log(self)

    def evaluate(self, graph: Pipeline) -> Fitness:
        outcome = self.evaluate_result(graph)
        if isinstance(outcome, EvaluationReused):
            outcome = outcome.original
        values = outcome.metrics if isinstance(outcome, EvaluationComplete) else None
        return to_fitness(values, self._objective.is_multi_objective)

    def evaluate_result(self, graph: Pipeline) -> EvaluationOutcome:
        graph.log = self._log
        self.last_outcome = None
        validation = self.validator(graph)
        if not validation.valid:
            return self._reject(graph, FailureKind.VALIDATION, '; '.join(validation.violations))

        evaluation_key = self._evaluation_key(graph)
        if self._do_unfit and evaluation_key in self._completed:
            self._completed.move_to_end(evaluation_key)
            cleanup = self._release(graph) if graph.is_fitted else None
            if cleanup is not None:
                return self._incomplete(graph, (), (), cleanup)
            self.last_outcome = EvaluationReused(self._completed[evaluation_key])
            return self.last_outcome

        attempts = []
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            metrics, failure, cleanup_failure = self._attempt(graph)
            attempts.append(AttemptRecord(0, attempt, failure, cleanup_failure))
            failure = cleanup_failure or failure
            if failure is None:
                record = FoldRecord(0, metrics, attempt, evaluation_key)
                outcome = EvaluationComplete(graph.descriptive_id, 1, (record,), tuple(attempts))
                if self._do_unfit and self.result_cache_size:
                    self._completed[evaluation_key] = outcome
                    while len(self._completed) > self.result_cache_size:
                        self._completed.popitem(last=False)
                self.last_outcome = outcome
                return outcome
            if not should_retry(self.retry_policy, failure, attempt):
                return self._incomplete(graph, (), tuple(attempts), failure)

        raise RuntimeError('retry policy exhausted without an evaluation outcome')

    def _attempt(self, graph: Pipeline):
        metrics = ()
        failure = cleanup_failure = None
        phase = FailureKind.FIT
        try:
            input_data = tensordata_to_input_data(self._tensor_data)
            input_data.supplementary_data.is_auto_preprocessed = True
            executor = PipelineOOFExecutor(
                pipeline=graph,
                input_data=input_data,
                cv_folds=self._cv_folds,
                n_jobs=self._eval_n_jobs,
                random_seed=self._random_seed,
                time_constraint=self._time_constraint,
            )
            phase = FailureKind.METRIC
            metrics = self._metric_values(graph, input_data, executor)
            if not metrics or not all(map(isfinite, metrics)):
                failure = EvaluationFailure(
                    FailureKind.METRIC, 'objective returned invalid or nonfinite fitness')
        except EXPECTED_ERRORS as error:
            failure = failure_from_exception(error, phase)
        finally:
            if self._do_unfit or failure is not None:
                cleanup_failure = self._release(graph)
        return metrics, failure, cleanup_failure

    def _metric_values(self, graph, input_data, executor):
        mode_outputs: Dict[str, object] = {}
        values = []
        for _, metric_func in self._objective.metrics:
            metric_cls = getattr(metric_func, '__self__', None)
            if isinstance(metric_cls, type) and issubclass(metric_cls, QualityMetric):
                output = mode_outputs.get(metric_cls.output_mode)
                if output is None:
                    output = executor.predict_oof(output_mode=metric_cls.output_mode)
                    mode_outputs[metric_cls.output_mode] = output
                metric_data = build_oof_metric_data(input_data, output)
                value = metric_cls.metric(metric_data.reference, metric_data.predicted)
                if metric_func.__name__ == 'get_value_with_penalty':
                    complexity = StructuralComplexity.get_value(graph)
                    penalty = abs(complexity * value * metric_cls.max_penalty_part)
                    value += min(penalty, abs(value * metric_cls.max_penalty_part))
            elif isinstance(metric_cls, type) and issubclass(metric_cls, ComplexityMetric):
                value = metric_func(graph)
            else:
                raise ValueError('OOF pipeline evaluation supports only FEDOT repository metrics.')
            values.append(float(value))
        return tuple(values)

    def _evaluation_key(self, graph):
        from fedot.extensions.registry import registered_extensions_identity

        return stable_hash((
            self.cache_namespace,
            graph.descriptive_id,
            tensor_data_identity(self._tensor_data),
            self._cv_folds,
            self._random_seed,
            tuple(self._objective.metric_names),
            self._eval_n_jobs,
            str(self._time_constraint),
            self._objective.is_multi_objective,
            registered_extensions_identity(),
        ))

    def _reject(self, graph, kind, message):
        failure = EvaluationFailure(kind, message)
        if graph.is_fitted:
            failure = self._release(graph) or failure
        return self._incomplete(graph, (), (), failure)

    def _incomplete(self, graph, folds, attempts, failure):
        self.last_outcome = EvaluationIncomplete(
            graph.descriptive_id, 1, folds, attempts, failure)
        return self.last_outcome

    @staticmethod
    def _release(graph):
        try:
            graph.unfit()
        except Exception as error:
            return EvaluationFailure(FailureKind.CLEANUP, str(error), type(error).__name__)
        return None

    def evaluate_intermediate_metrics(self, graph: Pipeline):
        """Intermediate-node metrics are not part of the OOF contract."""

    def clear_results(self):
        self._completed.clear()
        self.last_outcome = None
