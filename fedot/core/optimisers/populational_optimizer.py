import logging
from abc import abstractmethod
from typing import Any, Optional, Sequence

from tqdm import tqdm

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.grouped_condition import GroupedCondition


class PopulationalOptimizer(GraphOptimizer):
    """
    Base class of populational optimizer.
    PopulationalOptimizer implements all basic methods for optimization not related to evolution process
    to experiment with other kinds of evolution optimization methods
    It allows to find the optimal solution using specified metric (one or several).
    To implement the specific evolution strategy,
    the abstract method '_evolution_process' should be re-defined in the ancestor class

    :param objective: objective for optimization
    :param initial_graphs: graphs which were initialized outside the optimizer
    :param requirements: implementation-independent requirements for graph optimizer
    :param graph_generation_params: parameters for new graph generation
    :param parameters: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Pipeline],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional['GPGraphOptimizerParameters'] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, parameters)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.eval_dispatcher = MultiprocessingDispatcher(graph_adapter=graph_generation_params.adapter,
                                                         timer=self.timer,
                                                         n_jobs=requirements.n_jobs,
                                                         graph_cleanup_fn=_unfit_pipeline)

        # stopping_after_n_generation may be None, so use some obvious max number
        max_stagnation_length = parameters.stopping_after_n_generation or requirements.num_of_generations
        self.stop_optimisation = \
            GroupedCondition().add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.current_generation_num >= requirements.num_of_generations + 1,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_duration >= max_stagnation_length,
                'Optimisation finished: Early stopping criteria was satisfied'
            )

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_evaluation_callback(callback)

    def optimise(self, objective: ObjectiveFunction,
                 show_progress: bool = True) -> Sequence[OptGraph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective)

        with self.timer, tqdm(total=self.requirements.num_of_generations,
                              desc='Generations', unit='gen', initial=1,
                              disable=not show_progress or self.log.logging_level == logging.NOTSET):

            self._initial_population(evaluator=evaluator)

            while not self.stop_optimisation():
                new_population = self._evolve_population(evaluator=evaluator)
                # Adding of new population to history
                self._update_population(new_population)

        all_best_graphs = [ind.graph for ind in self.generations.best_individuals]
        return all_best_graphs

    @abstractmethod
    def _initial_population(self, *args, **kwargs):
        """ Initializes the initial population """
        raise NotImplementedError()

    @abstractmethod
    def _evolve_population(self, *args, **kwargs) -> PopulationT:
        """ Method realizing full evolution cycle """
        raise NotImplementedError()

    def _update_population(self, next_population: PopulationT):
        self._update_native_generation_numbers(next_population)
        self.generations.append(next_population)
        self._optimisation_callback(next_population, self.generations)
        self.population = next_population

        self.log.info(f'Generation num: {self.current_generation_num}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        self.log.info(f'no improvements for {self.generations.stagnation_duration} iterations')
        self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _update_native_generation_numbers(self, population: PopulationT):
        for individual in population:
            individual.set_native_generation(self.current_generation_num)

    def _progressbar(self, show_progress: bool = True):
        disable = not show_progress
        if disable:
            # disable call to tqdm.__init__ completely
            # to avoid access to stdout/stderr inside it
            # workaround for https://github.com/nccr-itmo/FEDOT/issues/765
            bar = EmptyProgressBar()
        else:
            bar = tqdm(total=self.requirements.num_of_generations,
                       desc='Generations', unit='gen', initial=1, disable=disable)
        return bar


def _unfit_pipeline(graph: Any):
    if isinstance(graph, Pipeline):
        graph.unfit()


class EmptyProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
