from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence

import datetime
from tqdm import tqdm

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import EvaluationOperator, PopulationT
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.utilities.grouped_condition import GroupedCondition


class PopulationalOptimizer(GraphOptimizer):
    """
    Base class of populational optimizer.
    PopulationalOptimizer implements all basic methods for optimization not related to evolution process
    to experiment with other kinds of evolution optimization methods
    It allows to find the optimal solution using specified metric (one or several).
    To implement the specific evolution strategy, implement `_evolution_process`.

    Args:
         objective: objective for optimization
         initial_graphs: graphs which were initialized outside the optimizer
         requirements: implementation-independent requirements for graph optimizer
         graph_generation_params: parameters for new graph generation
         graph_optimizer_params: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['GraphOptimizerParameters'] = None,
                 ):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.eval_dispatcher = MultiprocessingDispatcher(adapter=graph_generation_params.adapter,
                                                         n_jobs=requirements.n_jobs,
                                                         graph_cleanup_fn=_unfit_pipeline,
                                                         delegate_evaluator=graph_generation_params.remote_evaluator)

        # early_stopping_iterations and early_stopping_timeout may be None, so use some obvious max number
        max_stagnation_length = requirements.early_stopping_iterations or requirements.num_of_generations
        max_stagnation_time = requirements.early_stopping_timeout or self.timer.timeout
        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.requirements.num_of_generations is not None and
                        self.current_generation_num >= self.requirements.num_of_generations + 1,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_iter_count >= max_stagnation_length,
                'Optimisation finished: Early stopping iterations criteria was satisfied'
            ).add_condition(
                lambda: self.generations.stagnation_time_duration >= max_stagnation_time,
                'Optimisation finished: Early stopping timeout criteria was satisfied'
            )

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_evaluation_callback(callback)

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective, self.timer)

        with self.timer, self._progressbar:

            self._initial_population(evaluator)

            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    return [ind.graph for ind in self.best_individuals]
                # Adding of new population to history
                self._update_population(new_population)
        self._update_population(self.best_individuals, 'final_choices')
        return [ind.graph for ind in self.best_individuals]

    @property
    def best_individuals(self):
        return self.generations.best_individuals

    @abstractmethod
    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        raise NotImplementedError()

    @abstractmethod
    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """
        raise NotImplementedError()

    def _update_population(self, next_population: PopulationT, label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        self._optimisation_callback(next_population, self.generations)
        self._log_to_history(next_population, label, metadata)
        self.population = next_population

        self.log.info(f'Generation num: {self.current_generation_num}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        if self.generations.stagnation_iter_count > 0:
            self.log.info(f'no improvements for {self.generations.stagnation_iter_count} iterations')
            self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _log_to_history(self, population: PopulationT, label: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        self.history.add_to_history(population, label, metadata)
        self.history.add_to_archive_history(self.generations.best_individuals)
        if self.requirements.history_dir:
            self.history.save_current_results()

    @property
    def _progressbar(self):
        if self.requirements.show_progress:
            bar = tqdm(total=self.requirements.num_of_generations,
                       desc='Generations', unit='gen', initial=1)
        else:
            # disable call to tqdm.__init__ to avoid stdout/stderr access inside it
            # part of a workaround for https://github.com/nccr-itmo/FEDOT/issues/765
            bar = EmptyProgressBar()
        return bar


# TODO: remove this hack (e.g. provide smth like FitGraph with fit/unfit interface)
def _unfit_pipeline(graph: Any):
    if hasattr(graph, 'unfit'):
        graph.unfit()


class EmptyProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class EvaluationAttemptsError(Exception):
    """ Number of evaluation attempts exceeded """

    def __init__(self, *args):
        self.message = args[0] or None

    def __str__(self):
        return self.message or 'Too many fitness evaluation errors.'
