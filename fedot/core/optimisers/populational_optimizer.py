from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence

from tqdm import tqdm

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.grouped_condition import GroupedCondition

if TYPE_CHECKING:
    pass


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
    :param graph_optimizer_params: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['GraphOptimizerParameters'] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.eval_dispatcher = MultiprocessingDispatcher(graph_adapter=graph_generation_params.adapter,
                                                         timer=self.timer,
                                                         n_jobs=requirements.n_jobs,
                                                         graph_cleanup_fn=_unfit_pipeline)

        # stopping_after_n_generation may be None, so use some obvious max number
        max_stagnation_length = requirements.stopping_after_n_generation or requirements.num_of_generations
        self.stop_optimization = \
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

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective)

        with self.timer, self._progressbar:

            self._initial_population(evaluator=evaluator)

            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(evaluator=evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    return self.best_graphs
                # Adding of new population to history
                self._update_population(new_population)

        return self.best_graphs

    @property
    def best_graphs(self):
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


def _unfit_pipeline(graph: Any):
    if isinstance(graph, Pipeline):
        graph.unfit()


class EmptyProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class EvaluationAttemptsError(Exception):
    """ Number of evaluation attempts exceeded """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Too many fitness evaluation errors.'
