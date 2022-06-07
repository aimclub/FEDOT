import gc
import multiprocessing
import timeit
from abc import ABC, abstractmethod
from contextlib import closing
from random import choice
from typing import Dict, Optional

from fedot.core.dag.graph import Graph
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import EvaluationOperator, PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.timer import Timer, get_forever_timer
from fedot.remote.remote_evaluator import RemoteEvaluator


class ObjectiveEvaluationDispatcher(ABC):
    """Builder for evaluation operator.
    Takes objective function and decides how to evaluate it over population:
    - defines implementation-specific evaluation policy (e.g. sequential, parallel, async);
    - saves additional metadata (e.g. computation time, intermediate metrics values).
    """

    @abstractmethod
    def dispatch(self, objective: ObjectiveFunction) -> EvaluationOperator:
        """Return mapped objective function for evaluating population."""
        raise NotImplementedError()

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        """Set or reset (with None) post-evaluation callback
        that's called on each graph after its evaluation."""
        pass


class MultiprocessingDispatcher(ObjectiveEvaluationDispatcher):
    """Evaluates objective function on population using multiprocessing pool
    and optionally model evaluation cache with RemoteEvaluator.

    Usage: call `dispatch(objective_function)` to get evaluation function.

    :param graph_adapter: adapter for mapping between OptGraph and Graph.
    :param log: logger to use
    :param n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
    :param graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
    """

    def __init__(self,
                 graph_adapter: BaseOptimizationAdapter,
                 timer: Timer = None,
                 log: Log = None,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None):
        self._objective_eval = None
        self._graph_adapter = graph_adapter
        self._cleanup = graph_cleanup_fn
        self._post_eval_callback = None

        self.timer = timer or get_forever_timer()
        self.logger = log or default_log(self.__class__.__name__)
        self._n_jobs = n_jobs
        self._reset_eval_cache()

    def dispatch(self, objective: ObjectiveFunction) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
        return self.evaluate_with_cache

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        self._post_eval_callback = callback

    def evaluate_with_cache(self, population: PopulationT) -> PopulationT:
        reversed_population = list(reversed(population))
        self._remote_compute_cache(reversed_population)
        evaluated_population = self.evaluate_population(reversed_population)
        self._reset_eval_cache()
        if not evaluated_population and reversed_population:
            raise AttributeError('Too many fitness evaluation errors. Composing stopped.')
        return evaluated_population

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        if n_jobs == 1:
            mapped_evals = map(self.evaluate_single, individuals)
        else:
            with closing(multiprocessing.Pool(n_jobs)) as pool:
                mapped_evals = list(pool.imap_unordered(self.evaluate_single, individuals))

        # If there were no successful evals then try once again getting at least one,
        #  even if time limit was reached
        successful_evals = list(filter(None, mapped_evals))
        if not successful_evals:
            single = self.evaluate_single(choice(individuals), with_time_limit=False)
            if single:
                successful_evals = [single]

        return successful_evals

    def evaluate_single(self, ind: Individual, with_time_limit=True) -> Optional[Individual]:
        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        start_time = timeit.default_timer()

        graph = self.evaluation_cache.get(ind.uid, ind.graph)
        _restrict_n_jobs_in_nodes(graph)
        adapted_graph = self._graph_adapter.restore(graph)

        ind.fitness = self._objective_eval(adapted_graph)

        if self._post_eval_callback:
            self._post_eval_callback(adapted_graph)
        if self._cleanup:
            self._cleanup(adapted_graph)
        gc.collect()

        ind.graph = self._graph_adapter.adapt(adapted_graph)

        end_time = timeit.default_timer()
        ind.metadata['computation_time_in_seconds'] = end_time - start_time
        return ind if ind.fitness.valid else None

    def _reset_eval_cache(self):
        self.evaluation_cache: Dict[int, Graph] = {}

    def _remote_compute_cache(self, population: PopulationT):
        self._reset_eval_cache()
        fitter = RemoteEvaluator()  # singleton
        if fitter.use_remote:
            self.logger.info('Remote fit used')
            restored_graphs = [self._graph_adapter.restore(ind.graph) for ind in population]
            computed_pipelines = fitter.compute_pipelines(restored_graphs)
            self.evaluation_cache = {ind.uid: graph for ind, graph in zip(population, computed_pipelines)}


def determine_n_jobs(n_jobs=-1, logger=None):
    if n_jobs > multiprocessing.cpu_count() or n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs


def _restrict_n_jobs_in_nodes(graph: OptGraph):
    """ Function to prevent memory overflow due to many processes running in time"""
    for node in graph.nodes:
        if 'n_jobs' in node.content['params']:
            node.content['params']['n_jobs'] = 1
        if 'num_threads' in node.content['params']:
            node.content['params']['num_threads'] = 1
