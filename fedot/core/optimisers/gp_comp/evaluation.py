import gc
import multiprocessing
import timeit
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from random import choice
from typing import Dict, Optional

from joblib import Parallel, delayed

from fedot.core.dag.graph import Graph
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import EvaluationOperator, PopulationT
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.timer import Timer, get_forever_timer
from fedot.core.pipelines.verification import verifier_for_task
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
    :param n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
    :param graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
    """

    def __init__(self,
                 graph_adapter: BaseOptimizationAdapter,
                 timer: Timer = None,
                 n_jobs: int = 1,
                 sync_logs: bool = False,
                 graph_cleanup_fn: Optional[GraphFunction] = None):
        self._objective_eval = None
        self._graph_adapter = graph_adapter
        self._cleanup = graph_cleanup_fn
        self._post_eval_callback = None

        self.timer = timer or get_forever_timer()
        self.logger = default_log(self)
        self._n_jobs = n_jobs
        self._sync_logs = sync_logs
        self._reset_eval_cache()

    def dispatch(self, objective: ObjectiveFunction) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
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
        n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")
        logger_lvl = Log().logger.level
        if self._sync_logs:
            with Log.using_mp_listener() as shared_q:
                eval_inds = parallel(delayed(self.evaluate_single)(ind=ind, logs_queue=shared_q, logs_lvl=logger_lvl)
                                     for ind in individuals)
        else:
            eval_inds = parallel(delayed(self.evaluate_single)(ind=ind, logs_lvl=logger_lvl)
                                 for ind in individuals)
        # If there were no successful evals then try once again getting at least one,
        # even if time limit was reached
        successful_evals = list(filter(None, eval_inds))
        if not successful_evals:
            single = self.evaluate_single(choice(individuals), with_time_limit=False)
            if single:
                successful_evals = [single]
            else:
                successful_evals = None

        return successful_evals

    def evaluate_single(self, ind: Individual, with_time_limit: bool = True, logs_lvl: Optional[int] = None,
                        logs_queue: Optional[multiprocessing.Queue] = None) -> Optional[Individual]:
        if ind.fitness.valid:
            return ind
        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_lvl is not None:
            # in case of multiprocessing run
            Log().reset_logging_level(logs_lvl)
        if logs_queue is not None:
            # in case of multiprocessing run
            logger_context = Log.using_mp_worker(logs_queue)
        else:
            logger_context = nullcontext()
        with logger_context:
            start_time = timeit.default_timer()

            graph = self.evaluation_cache.get(ind.uid, ind.graph)
            adapted_graph = self._graph_adapter.restore(graph)

            ind_fitness = self._objective_eval(adapted_graph)
            if self._post_eval_callback:
                self._post_eval_callback(adapted_graph)
            if self._cleanup:
                self._cleanup(adapted_graph)
            gc.collect()

            ind_graph = self._graph_adapter.adapt(adapted_graph)

            ind.set_evaluation_result(ind_fitness, ind_graph)

            end_time = timeit.default_timer()
            ind.metadata['computation_time_in_seconds'] = end_time - start_time
            ind.metadata['evaluation_time_iso'] = datetime.now().isoformat()
            return ind if ind.fitness.valid else None

    def _reset_eval_cache(self):
        self.evaluation_cache: Dict[str, Graph] = {}

    def _remote_compute_cache(self, population: PopulationT):
        self._reset_eval_cache()
        fitter = RemoteEvaluator()  # singleton
        if fitter.use_remote:
            self.logger.info('Remote fit used')
            restored_graphs = [self._graph_adapter.restore(ind.graph) for ind in population]
            verifier = verifier_for_task(task_type=None, adapter=self._graph_adapter)
            computed_pipelines = fitter.compute_graphs(restored_graphs, verifier)
            self.evaluation_cache = {ind.uid: graph for ind, graph in zip(population, computed_pipelines)}


class SimpleDispatcher(ObjectiveEvaluationDispatcher):
    """Evaluates objective function on population.
    Usage: call `dispatch(objective_function)` to get evaluation function.
    :param graph_adapter: adapter for mapping between OptGraph and Graph.
    :param timer: timer to set timeout for evaluation of population
    """

    def __init__(self,
                 graph_adapter: BaseOptimizationAdapter,
                 timer: Timer = None):
        self._objective_eval = None
        self._graph_adapter = graph_adapter
        self.timer = timer or get_forever_timer()

    def dispatch(self, objective: ObjectiveFunction) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
        return self.evaluate_population

    def evaluate_population(self, individuals: PopulationT) -> Optional[PopulationT]:
        mapped_evals = list(map(self.evaluate_single, individuals))
        evaluated_population = list(filter(None, mapped_evals))
        if not evaluated_population:
            evaluated_population = None
        return evaluated_population

    def evaluate_single(self, ind: Individual, with_time_limit=True) -> Optional[Individual]:
        if ind.fitness.valid:
            return ind
        if with_time_limit and self.timer.is_time_limit_reached():
            return None

        start_time = timeit.default_timer()

        graph = ind.graph
        adapted_graph = self._graph_adapter.restore(graph)
        ind_fitness = self._objective_eval(adapted_graph)
        ind_graph = self._graph_adapter.adapt(adapted_graph)
        ind.set_evaluation_result(ind_fitness, ind_graph)
        end_time = timeit.default_timer()
        ind.metadata['computation_time_in_seconds'] = end_time - start_time
        ind.metadata['evaluation_time_iso'] = datetime.now().isoformat()
        return ind if ind.fitness.valid else None


def determine_n_jobs(n_jobs=-1, logger=None):
    if n_jobs > multiprocessing.cpu_count() or n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs
