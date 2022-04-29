import gc
import multiprocessing
import timeit
from contextlib import closing
from random import choice

from typing import Dict, Optional

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.gp_comp.operators.operator import *
from fedot.core.optimisers.timer import Timer, get_forever_timer
from fedot.core.validation.objective_eval import ObjectiveEvaluate
from fedot.remote.remote_evaluator import RemoteEvaluator


G = TypeVar('G', bound=Graph)
GN = TypeVar('GN', bound=GraphNode)


class EvaluationDispatcher(Generic[G, GN], Operator[PopulationT]):
    """Defines objective-independent details of how evaluation of individuals must be handled.
    Responsibilities:
    - Handle evaluation policy (e.g. sequential, parallel, async) and dispatch evaluation.
    - Delegate Fitness computation to ObjectiveEvaluate for each individual in the population.
    - Save additional metadata related to evaluation process (e.g. computation time)
    """
    def __init__(self,
                 objective_eval: ObjectiveEvaluate[G],
                 graph_adapter: BaseOptimizationAdapter[G, GN],
                 timer: Timer = None,
                 log: Log = None,
                 n_jobs: int = 1,
                 collect_intermediate_metrics: bool = False):
        self._objective_eval = objective_eval
        self._graph_adapter = graph_adapter
        self._collect_intermediate_metrics = collect_intermediate_metrics

        self.timer = timer or get_forever_timer()
        self.logger = log or default_log('Population evaluation')
        self._n_jobs = n_jobs
        self._reset_eval_cache()

    def __call__(self, population: PopulationT) -> PopulationT:
        reversed_population = list(reversed(population))
        self._remote_compute_cache(reversed_population)
        evaluated_population = self.evaluate_dispatch(reversed_population)
        self._reset_eval_cache()
        if not evaluated_population and reversed_population:
            raise AttributeError('Too many fitness evaluation errors. Composing stopped.')
        return evaluated_population

    def evaluate_dispatch(self, individuals: PopulationT) -> PopulationT:
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
        if self._collect_intermediate_metrics:
            self._objective_eval.evaluate_intermediate_metrics(adapted_graph)
        self._cleanup_memory(adapted_graph)
        ind.graph = self._graph_adapter.adapt(adapted_graph)

        end_time = timeit.default_timer()
        ind.metadata['computation_time_in_seconds'] = end_time - start_time
        return ind if ind.fitness.valid else None

    def _cleanup_memory(self, graph: Graph):
        self._objective_eval.cleanup(graph)
        gc.collect()

    def _reset_eval_cache(self):
        self.evaluation_cache: Dict[int, G] = {}

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
