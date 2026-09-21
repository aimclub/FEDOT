"""Typed public composition hooks, implemented without GOLEM private methods."""
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Protocol

from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, SequentialDispatcher
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.fitness import Fitness

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.optimisers.objective.data_objective_eval import (
    PipelineObjectiveEvaluateWithTensorData, TensorDataSource,
)
from fedot.core.optimisers.objective.evaluation_contracts import (
    PipelineValidator, RetryPolicy, validate_pipeline,
)
from fedot.core.optimisers.population import (
    BoundedReproduction, ContractEvaluationDispatcher, ReproductionPolicy, UniqueEvaluationDispatcher,
)


@dataclass(frozen=True)
class EvaluationRequest:
    """Runtime inputs to an evaluator factory; not a serializable decision plan."""
    objective: Objective
    data_producer: TensorDataSource
    time_constraint: Optional[timedelta] = None
    validation_blocks: Optional[int] = None
    operations_cache: Optional[OperationsCache] = None
    predictions_cache: Optional[PredictionsCache] = None
    eval_n_jobs: int = 1
    retry_policy: RetryPolicy = RetryPolicy()
    validator: PipelineValidator = validate_pipeline
    expected_folds: Optional[int] = None
    cache_namespace: str = 'fedot-evaluation-v1'


class EvaluationService(Protocol):
    def evaluate(self, graph: Pipeline) -> Fitness:
        ...

    def evaluate_intermediate_metrics(self, graph: Pipeline) -> None:
        ...

    def clear_results(self) -> None:
        ...


class EvaluatorFactory(Protocol):
    def __call__(self, request: EvaluationRequest) -> EvaluationService:
        ...


def build_evaluator(request: EvaluationRequest) -> PipelineObjectiveEvaluateWithTensorData:
    return PipelineObjectiveEvaluateWithTensorData(
        request.objective, request.data_producer, time_constraint=request.time_constraint,
        validation_blocks=request.validation_blocks, operations_cache=request.operations_cache,
        predictions_cache=request.predictions_cache, eval_n_jobs=request.eval_n_jobs,
        retry_policy=request.retry_policy, validator=request.validator,
        expected_folds=request.expected_folds, cache_namespace=request.cache_namespace)


@dataclass(frozen=True)
class EvolutionHooks:
    """Execution capabilities; not a serializable pure decision plan."""
    evaluator_factory: EvaluatorFactory = build_evaluator
    validator: PipelineValidator = validate_pipeline
    retry_policy: RetryPolicy = RetryPolicy()
    reproduction: Optional[ReproductionPolicy] = None
    expected_folds: Optional[int] = None
    cache_namespace: str = 'fedot-evaluation-v1'

    def __post_init__(self):
        if not callable(self.evaluator_factory) or not callable(self.validator):
            raise TypeError('evaluator_factory and validator must be callable')
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError('retry_policy must be RetryPolicy')
        if self.reproduction is not None and not isinstance(self.reproduction, ReproductionPolicy):
            raise TypeError('reproduction must be ReproductionPolicy or None')
        if self.expected_folds is not None and (type(self.expected_folds) is not int or self.expected_folds < 1):
            raise ValueError('expected_folds must be a positive integer or None')
        if not isinstance(self.cache_namespace, str) or not self.cache_namespace.strip():
            raise ValueError('cache_namespace must be a nonempty string')


def configure_golem_evaluation(optimizer, hooks: EvolutionHooks):
    """Install public-API adapters before optimise(), never a second optimizer.

    GOLEM has no setter for a reproducer/dispatcher factory. Its existing public
    attributes are the narrow compatibility boundary, checked here explicitly.
    """
    dispatcher = getattr(optimizer, 'eval_dispatcher', None)
    if isinstance(dispatcher, UniqueEvaluationDispatcher):
        return
    if type(dispatcher) not in (MultiprocessingDispatcher, SequentialDispatcher):
        if hooks.reproduction is not None:
            raise TypeError('custom optimizer must provide its own typed reproduction/evaluation adapter')
        return
    params = optimizer.graph_generation_params
    # FEDOT owns attempt cleanup; disable GOLEM's second unfit call via its
    # constructor API instead of patching the private _cleanup attribute.
    jobs = optimizer.requirements.n_jobs if type(dispatcher) is MultiprocessingDispatcher else 1
    delegate = ContractEvaluationDispatcher(adapter=params.adapter, n_jobs=jobs,
                                            graph_cleanup_fn=None, delegate_evaluator=params.remote_evaluator)
    optimizer.eval_dispatcher = UniqueEvaluationDispatcher(delegate)
    if hooks.reproduction is not None:
        if not hasattr(optimizer.reproducer, 'reproduce_uncontrolled'):
            raise TypeError('GOLEM reproducer must expose reproduce_uncontrolled')
        optimizer.reproducer = BoundedReproduction(
            optimizer.reproducer, hooks.reproduction, lambda: optimizer.graph_optimizer_params.pop_size)
