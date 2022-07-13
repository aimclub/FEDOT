import logging
from copy import deepcopy
from random import choice
from typing import Any, List, Optional, Sequence, Union

from tqdm import tqdm

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.constants import MAXIMAL_ATTEMPTS_NUMBER
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, crossover_parents_selection, selection
from fedot.core.optimisers.gp_comp.parameters.graph_depth import AdaptiveGraphDepth
from fedot.core.optimisers.gp_comp.parameters.operators_prob import init_adaptive_operators_prob
from fedot.core.optimisers.gp_comp.parameters.population_size import PopulationSize, init_adaptive_pop_size
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.grouped_condition import GroupedCondition


class GPGraphOptimiserParameters(GraphOptimiserParameters):
    """
    This class is for defining the parameters of optimiser

    :param selection_types: List of selection operators types
    :param crossover_types: List of crossover operators types
    :param mutation_types: List of mutation operators types
    :param regularization_type: type of regularization operator
    :param genetic_scheme_type: type of genetic evolutionary scheme
    :param elitism_type: type of elitism operator
    evolution. Default False.
    """

    def set_default_params(self):
        """
        Choose default configuration of the evolutionary operators
        """
        if not self.selection_types:
            if self.multi_objective:
                self.selection_types = [SelectionTypesEnum.spea2]
            else:
                self.selection_types = [SelectionTypesEnum.tournament]

        if not self.crossover_types:
            self.crossover_types = [CrossoverTypesEnum.subtree, CrossoverTypesEnum.one_point]

        if not self.mutation_types:
            # default mutation types
            self.mutation_types = [MutationTypesEnum.simple,
                                   MutationTypesEnum.reduce,
                                   MutationTypesEnum.growth,
                                   MutationTypesEnum.local_growth]

    def __init__(self, selection_types: List[SelectionTypesEnum] = None,
                 crossover_types: List[Union[CrossoverTypesEnum, Any]] = None,
                 mutation_types: List[Union[MutationTypesEnum, Any]] = None,
                 regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none,
                 genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational,
                 elitism_type: ElitismTypesEnum = ElitismTypesEnum.none,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.elitism_type = elitism_type

        self.set_default_params()  # always initialize in proper state


class EvoGraphOptimiser(GraphOptimiser):
    """
    Multi-objective evolutionary graph optimiser named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Pipeline],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimiserParameters] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, parameters)
        self.parameters = parameters or GPGraphOptimiserParameters()
        self.elitism = Elitism(self.parameters.elitism_type)
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

        # Define adaptive parameters

        self._pop_size: PopulationSize = \
            init_adaptive_pop_size(parameters.genetic_scheme_type, requirements, self.generations)

        self._operators_prob = \
            init_adaptive_operators_prob(parameters.genetic_scheme_type, requirements)

        # Define parameters

        self._min_population_size_with_elitism = 5

        start_depth = requirements.start_depth or requirements.max_depth
        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=start_depth,
                                               max_depth=requirements.max_depth,
                                               max_stagnated_generations=parameters.depth_increase_step,
                                               adaptive=parameters.with_auto_depth_configuration)
        self.max_depth = self._graph_depth.initial

        self.initial_individuals = \
            [Individual(self.graph_generation_params.adapter.adapt(graph)) for graph in initial_graphs]

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def _extend_population(self, initial_individuals, pop_size: int, max_depth: int):
        iter_num = 0
        initial_graphs = [ind.graph for ind in self.initial_individuals]
        while len(initial_individuals) < pop_size:
            initial_req = deepcopy(self.requirements)
            initial_req.mutation_prob = 1

            new_ind = self._mutate(choice(self.initial_individuals), max_depth, custom_requirements=initial_req)
            new_graph = new_ind.graph
            iter_num += 1
            if new_graph not in initial_graphs and self.graph_generation_params.verifier(new_graph):
                initial_individuals.append(new_ind)
                initial_graphs.append(new_graph)
            if iter_num > MAXIMAL_ATTEMPTS_NUMBER:
                self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping.'
                                 f'Current size {len(self.initial_individuals)} instead of {pop_size} graphs.')
                break
        return initial_individuals

    def _next_population(self, next_population: PopulationT):
        self.update_native_generation_numbers(next_population)
        self.generations.append(next_population)
        self._optimisation_callback(next_population, self.generations)
        self.population = next_population
        self._operators_prob_update()

        self.log.info(f'Generation num: {self.current_generation_num}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        self.log.info(f'no improvements for {self.generations.stagnation_duration} iterations')
        self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _operators_prob_update(self):
        if not self.generations.is_any_improved:
            self.requirements.mutation_prob, self.requirements.crossover_prob = \
                self._operators_prob.next(self.population)

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_evaluation_callback(callback)

    def update_native_generation_numbers(self, population: PopulationT):
        for individual in population:
            individual.set_native_generation(self.current_generation_num)

    def optimise(self, objective: ObjectiveFunction,
                 show_progress: bool = True) -> Sequence[OptGraph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective)

        with self.timer, tqdm(total=self.requirements.num_of_generations,
                              desc='Generations', unit='gen', initial=1,
                              disable=not show_progress or self.log.verbosity_level == logging.NOTSET):
            # Adding of initial assumptions to history as zero generation
            self._next_population(evaluator(self.initial_individuals))
            pop_size = self._pop_size.initial
            if len(self.initial_individuals) < pop_size:
                self.initial_individuals = self._extend_population(self.initial_individuals, pop_size, self.max_depth)
                # Adding of extended population to history as zero generation
                self._next_population(evaluator(self.initial_individuals))

            while not self.stop_optimisation():
                pop_size = self._pop_size.next(self.population)
                self.max_depth = self._graph_depth.next()
                self.log.info(f'Next population size: {pop_size}; max graph depth: {self.max_depth}')

                individuals_to_select = regularized_population(self.parameters.regularization_type,
                                                               self.population,
                                                               evaluator,
                                                               self.graph_generation_params)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=pop_size,
                                                 params=self.graph_generation_params)
                new_population = self._reproduce(selected_individuals)

                new_population = evaluator(new_population)

                new_population = self._inheritance(new_population, pop_size)
                # Adding of new population to history
                self._next_population(new_population)
        all_best_graphs = [ind.graph for ind in self.generations.best_individuals]
        return all_best_graphs

    def with_elitism(self, pop_size: int) -> bool:
        if self.objective.is_multi_objective:
            return False
        else:
            return pop_size >= self._min_population_size_with_elitism

    def _inheritance(self, offspring: PopulationT, pop_size: int) -> PopulationT:
        """Gather next population given new offspring and previous population.
        :param offspring: offspring of current population.
        :param pop_size: size of the next population.
        :return: next population."""

        # TODO: from inheritance together with elitism we can get duplicate inds!
        offspring = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                self.population,
                                offspring, pop_size,
                                graph_params=self.graph_generation_params)
        return offspring

    def _mutate(self, individual: Individual,
                max_depth: Optional[int] = None,
                custom_requirements: Optional[PipelineComposerRequirements] = None) -> Individual:
        max_depth = max_depth or self.max_depth
        requirements = custom_requirements or self.requirements
        return mutation(types=self.parameters.mutation_types, params=self.graph_generation_params,
                        individual=individual, requirements=requirements, max_depth=max_depth)

    def _reproduce(self, population: PopulationT) -> PopulationT:
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in crossover_parents_selection(population):
                new_population += self._crossover_pair(ind_1, ind_2)
        new_population = list(map(self._mutate, new_population))
        return new_population

    def _crossover_pair(self, individual1: Individual, individual2: Individual) -> Sequence[Individual]:
        return crossover(self.parameters.crossover_types, individual1, individual2, max_depth=self.max_depth,
                         crossover_prob=self.requirements.crossover_prob, params=self.graph_generation_params)


def _unfit_pipeline(graph: Any):
    if isinstance(graph, Pipeline):
        graph.unfit()
