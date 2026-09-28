import logging
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple

import dask
from golem.core.log import Log
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.timer import Timer
from golem.utilities.memory import MemoryAnalytics
from golem.utilities.utilities import determine_n_jobs
from joblib import wrap_non_picklable_objects
from pymonad.either import Either
from pymonad.maybe import Maybe

from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels


class IndustrialDispatcher(MultiprocessingDispatcher):

    def dispatch(self, objective: ObjectiveFunction,
                 timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        return self.evaluate_with_cache

    def _multithread_eval(self, individuals_to_evaluate):
        log = Log().get_parameters()
        evaluation_results = list(map(lambda ind:
                                      self.industrial_evaluate_single(self,
                                                                      graph=ind.graph,
                                                                      uid_of_individual=ind.uid,
                                                                      logs_initializer=log),
                                      individuals_to_evaluate))
        evaluation_results = dask.compute(*evaluation_results)
        return evaluation_results

    def _eval_at_least_one(self, individuals):
        successful_evals = None
        for single_ind in individuals:
            try:
                evaluation_result = self.industrial_evaluate_single(
                    self, graph=single_ind.graph, uid_of_individual=single_ind.uid, with_time_limit=False)
                successful_evals = self.apply_evaluation_results(
                    [single_ind], [evaluation_result])
                if successful_evals:
                    break
            except Exception:
                successful_evals = None
        return successful_evals

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)

        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        individuals_evaluated = Maybe(individuals,
                                      monoid=[individuals, True]).then(
            lambda generation: self._multithread_eval(generation)). \
            then(lambda eval_res: self.apply_evaluation_results(
                individuals_to_evaluate, eval_res)).value

        successful_evals = individuals_evaluated + individuals_to_skip
        self.population_evaluation_info(evaluated_pop_size=len(
            successful_evals), pop_size=len(individuals))
        successful_evals = Either(successful_evals,
                                  monoid=[individuals_evaluated, not successful_evals]).either(
            left_function=lambda x: x,
            right_function=lambda y: self._eval_at_least_one(y))

        MemoryAnalytics.log(self.logger, additional_info='parallel evaluation of population',
                            logging_level=logging.INFO)
        return successful_evals

    @dask.delayed
    def eval_ind(self, graph, uid_of_individual):
        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()
        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual,
            fitness=fitness,
            graph=graph,
            metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso})
        return eval_res

    @wrap_non_picklable_objects
    def industrial_evaluate_single(self,
                                   graph: OptGraph,
                                   uid_of_individual: str,
                                   with_time_limit: bool = True,
                                   cache_key: Optional[str] = None,
                                   logs_initializer: Optional[Tuple[int,
                                                                    pathlib.Path]] = None) -> GraphEvalResult:
        if self._n_jobs != 1:
            IndustrialModels().setup_repository()

        graph = self.evaluation_cache.get(cache_key, graph)
        #
        # if with_time_limit and self.timer.is_time_limit_reached():
        #     return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        return self.eval_ind(graph, uid_of_individual)
