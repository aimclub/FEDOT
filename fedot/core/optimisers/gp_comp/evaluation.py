import gc
import multiprocessing
import pathlib
import timeit
from abc import ABC, abstractmethod
from datetime import datetime
from random import choice
from typing import Dict, Optional, Tuple

from joblib import Parallel, delayed, cpu_count

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.adapter import BaseOptimizationAdapter
from fedot.core.optimisers.fitness import Fitness
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

    :param n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
    :param graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 timer: Timer = None,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None):
        self._adapter = adapter
        self._objective_eval = None
        self._cleanup = graph_cleanup_fn
        self._post_eval_callback = None

        self.timer = timer or get_forever_timer()
        self.logger = default_log(self)
        self._n_jobs = n_jobs
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
        eval_inds = parallel(delayed(self.evaluate_single)(ind=ind, logs_initializer=Log().get_parameters())
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

    def evaluate_single(self, ind: Individual, with_time_limit: bool = True,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> Optional[Individual]:
        if ind.fitness.valid:
            return ind
        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)
        start_time = timeit.default_timer()

        graph = self.evaluation_cache.get(ind.uid, ind.graph)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        ind_fitness, ind_domain_graph = adapted_evaluate(graph)
        ind.set_evaluation_result(ind_fitness, ind_domain_graph)

        end_time = timeit.default_timer()

        ind.metadata['computation_time_in_seconds'] = end_time - start_time
        ind.metadata['evaluation_time_iso'] = datetime.now().isoformat()
        return ind if ind.fitness.valid else None

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
        fitter = RemoteEvaluator()  # singleton
        if fitter.use_remote:
            self.logger.info('Remote fit used')
            restored_graphs = self._adapter.restore_population(population)
            verifier = verifier_for_task(task_type=None)
            computed_pipelines = fitter.compute_graphs(restored_graphs, verifier)
            self.evaluation_cache = {ind.uid: graph for ind, graph in zip(population, computed_pipelines)}


class SimpleDispatcher(ObjectiveEvaluationDispatcher):
    """Evaluates objective function on population.

    Usage: call `dispatch(objective_function)` to get evaluation function.

    :param timer: timer to set timeout for evaluation of population
    """

    def __init__(self, adapter: BaseOptimizationAdapter, timer: Timer = None):
        self._adapter = adapter
        self._objective_eval = None
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

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        ind_fitness, ind_graph = adapted_evaluate(ind.graph)
        ind.set_evaluation_result(ind_fitness, ind_graph)

        end_time = timeit.default_timer()

        ind.metadata['computation_time_in_seconds'] = end_time - start_time
        ind.metadata['evaluation_time_iso'] = datetime.now().isoformat()
        return ind if ind.fitness.valid else None

    def _evaluate_graph(self, graph: Graph) -> Tuple[Fitness, Graph]:
        fitness = self._objective_eval(graph)
        return fitness, graph


def determine_n_jobs(n_jobs=-1, logger=None):
    if n_jobs > cpu_count() or n_jobs == -1:
        n_jobs = cpu_count()
    if logger:
        logger.info(f"Number of used CPU's: {n_jobs}")
    return n_jobs
