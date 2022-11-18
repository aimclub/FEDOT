import gc
import pathlib
import timeit
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from random import choice
from typing import Dict, Optional, Tuple, List, TypeVar, Sequence

from joblib import Parallel, delayed, cpu_count

from fedot.core.adapter import BaseOptimizationAdapter
from fedot.core.dag.graph import Graph
from fedot.core.log import default_log, Log
from fedot.core.optimisers.fitness import Fitness
from fedot.core.optimisers.gp_comp.operators.operator import EvaluationOperator, PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.opt_history_objects.individual import GraphEvalResult
from fedot.core.optimisers.timer import Timer, get_forever_timer
from fedot.core.utilities.serializable import Serializable

OptionalEvalResult = Optional[GraphEvalResult]
EvalResultsList = List[OptionalEvalResult]
G = TypeVar('G', bound=Serializable)


class DelegateEvaluator:
    """Interface for delegate evaluator of graphs."""
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        return False

    @abstractmethod
    def compute_graphs(self, graphs: Sequence[G]) -> Sequence[G]:
        raise NotImplementedError()


class ObjectiveEvaluationDispatcher(ABC):
    """Builder for evaluation operator.
    Takes objective function and decides how to evaluate it over population:
    - defines implementation-specific evaluation policy (e.g. sequential, parallel, async);
    - saves additional metadata (e.g. computation time, intermediate metrics values).
    """

    @abstractmethod
    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return mapped objective function for evaluating population.

        Args:
            objective: objective function that accepts single individual
            timer: optional timer for stopping the evaluation process

        Returns:
            EvaluationOperator: objective function that accepts whole population
        """
        raise NotImplementedError()

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        """Set or reset (with None) post-evaluation callback
        that's called on each graph after its evaluation.

        Args:
            callback: callback to be called on each evaluated graph
        """
        pass

    @staticmethod
    def split_individuals_to_evaluate(individuals: PopulationT) -> Tuple[PopulationT, PopulationT]:
        """Split individuals sequence to evaluated and skipped ones."""
        individuals_to_evaluate = []
        individuals_to_skip = []
        for ind in individuals:
            if ind.fitness.valid:
                individuals_to_skip.append(ind)
            else:
                individuals_to_evaluate.append(ind)
        return individuals_to_evaluate, individuals_to_skip

    @staticmethod
    def apply_evaluation_results(individuals: PopulationT,
                                 evaluation_results: EvalResultsList) -> PopulationT:
        """Applies results of evaluation to the evaluated population.
        Excludes individuals that weren't evaluated."""
        evaluation_results = {res.uid_of_individual: res for res in evaluation_results if res}
        individuals_evaluated = []
        for ind in individuals:
            eval_res = evaluation_results.get(ind.uid)
            if not eval_res:
                continue
            ind.set_evaluation_result(eval_res)
            individuals_evaluated.append(ind)
        return individuals_evaluated


class MultiprocessingDispatcher(ObjectiveEvaluationDispatcher):
    """Evaluates objective function on population using multiprocessing pool
    and optionally model evaluation cache with RemoteEvaluator.

    Usage: call `dispatch(objective_function)` to get evaluation function.

    Args:
        adapter: adapter for graphs
        n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
        graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
        delegate_evaluator: delegate graph fitter (e.g. for remote graph fitting before evaluation)
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None,
                 delegate_evaluator: Optional[DelegateEvaluator] = None):
        self._adapter = adapter
        self._objective_eval = None
        self._cleanup = graph_cleanup_fn
        self._post_eval_callback = None
        self._delegate_evaluator = delegate_evaluator

        self.timer = None
        self.logger = default_log(self)
        self._n_jobs = n_jobs
        self._reset_eval_cache()

    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
        self.timer = timer or get_forever_timer()
        return self.evaluate_with_cache

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        self._post_eval_callback = callback

    def evaluate_with_cache(self, population: PopulationT) -> Optional[PopulationT]:
        reversed_population = list(reversed(population))
        self._remote_compute_cache(reversed_population)
        evaluated_population = self.evaluate_population(reversed_population)
        self._reset_eval_cache()
        return evaluated_population

    def evaluate_population(self, individuals: PopulationT) -> Optional[PopulationT]:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        # Evaluate individuals without valid fitness in parallel.
        n_jobs = determine_n_jobs(self._n_jobs, self.logger)
        parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")
        eval_func = partial(self.evaluate_single, logs_initializer=Log().get_parameters())
        evaluation_results = parallel(delayed(eval_func)(ind.graph, ind.uid) for ind in individuals_to_evaluate)
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        # If there were no successful evals then try once again getting at least one,
        # even if time limit was reached
        successful_evals = individuals_evaluated + individuals_to_skip
        if not successful_evals:
            single_ind = choice(individuals)
            evaluation_result = eval_func(single_ind.graph, single_ind.uid, with_time_limit=False)
            successful_evals = self.apply_evaluation_results([single_ind], [evaluation_result]) or None
        return successful_evals

    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit: bool = True, cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> OptionalEvalResult:

        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        graph = self.evaluation_cache.get(cache_key, graph)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso
            }
        )
        return eval_res

    def _evaluate_graph(self, domain_graph: Graph) -> Tuple[Fitness, Graph]:
        fitness = self._objective_eval(domain_graph)

        if self._post_eval_callback:
            self._post_eval_callback(domain_graph)
        if self._cleanup:
            self._cleanup(domain_graph)
        gc.collect()

        return fitness, domain_graph

    def _reset_eval_cache(self):
        self.evaluation_cache: Dict[str, Graph] = {}

    def _remote_compute_cache(self, population: PopulationT):
        self._reset_eval_cache()
        if self._delegate_evaluator and self._delegate_evaluator.is_enabled:
            self.logger.info('Remote fit used')
            restored_graphs = self._adapter.restore(population)
            computed_graphs = self._delegate_evaluator.compute_graphs(restored_graphs)
            self.evaluation_cache = {ind.uid: graph for ind, graph in zip(population, computed_graphs)}


class SimpleDispatcher(ObjectiveEvaluationDispatcher):
    """Evaluates objective function on population.

    Usage: call `dispatch(objective_function)` to get evaluation function.
    """

    def __init__(self, adapter: BaseOptimizationAdapter):
        self._adapter = adapter
        self._objective_eval = None
        self.timer = None

    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
        self.timer = timer or get_forever_timer()
        return self.evaluate_population

    def evaluate_population(self, individuals: PopulationT) -> Optional[PopulationT]:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        evaluation_results = [self.evaluate_single(ind.graph, ind.uid) for ind in individuals_to_evaluate]
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        evaluated_population = individuals_evaluated + individuals_to_skip or None
        return evaluated_population

    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit=True) -> OptionalEvalResult:
        if with_time_limit and self.timer.is_time_limit_reached():
            return None

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()

        eval_time_iso = datetime.now().isoformat()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso
            }
        )
        return eval_res

    def _evaluate_graph(self, graph: Graph) -> Tuple[Fitness, Graph]:
        fitness = self._objective_eval(graph)
        return fitness, graph


def determine_n_jobs(n_jobs=-1, logger=None):
    if n_jobs > cpu_count() or n_jobs == -1:
        n_jobs = cpu_count()
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs
