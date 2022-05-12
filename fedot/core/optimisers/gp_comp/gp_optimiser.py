from copy import deepcopy
from functools import partial
from itertools import zip_longest
from typing import Any, Optional, Union, List, Iterable, Sequence

from tqdm import tqdm

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import (
    clean_operators_history,
    num_of_parents_in_crossover,
    random_graph
)
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.initial_population_builder import InitialPopulationBuilder
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.operators.evaluation import EvaluationDispatcher
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.generation_keeper import GenerationKeeper
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.parameters.graph_depth import GraphDepth
from fedot.core.optimisers.gp_comp.parameters.population_size import PopulationSize, ConstRatePopulationSize
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, selection
from fedot.core.utilities.grouped_condition import GroupedCondition
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate


class GPGraphOptimiserParameters(GraphOptimiserParameters):
    """
        This class is for defining the parameters of optimiser

        :param selection_types: List of selection operators types
        :param crossover_types: List of crossover operators types
        :param mutation_types: List of mutation operators types
        :param regularization_type: type of regularization operator
        :param genetic_scheme_type: type of genetic evolutionary scheme
        :param with_auto_depth_configuration: flag to enable option of automated tree depth configuration during
        evolution. Default False.
        :param depth_increase_step: the step of depth increase in automated depth configuration
        :param multi_objective: flag used for algorithm of type definition (multi-objective if true and single-objective
        if false). Value is defined in ComposerBuilder. Default False.
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
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type

        self.set_default_params()  # always initialize in proper state


class EvoGraphOptimiser(GraphOptimiser):
    """
    Multi-objective evolutionary graph optimiser named GPComp
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 objective: Objective,
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 log: Optional[Log] = None):
        super().__init__(initial_graph, objective, requirements, graph_generation_params, parameters, log)

        self.graph_generation_params = graph_generation_params
        self.requirements = requirements
        self._min_population_size_with_elitism = 3

        self.parameters = parameters or GPGraphOptimiserParameters()

        is_steady_state = self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state
        self._pop_size: PopulationSize = ConstRatePopulationSize(
            pop_size=requirements.pop_size or 10,
            offspring_rate=1.0 if is_steady_state else requirements.offspring_rate
        )

        self.population = None
        self.generations = GenerationKeeper(self.objective)

        start_depth = requirements.start_depth or requirements.max_depth
        self._graph_depth = GraphDepth(self.generations,
                                       start_depth=start_depth,
                                       max_depth=requirements.max_depth,
                                       max_stagnated_generations=parameters.depth_increase_step,
                                       adaptive=parameters.with_auto_depth_configuration)
        self.max_depth = self._graph_depth.initial

        self.timer = OptimisationTimer(timeout=self.requirements.timeout, log=self.log)

        # stopping_after_n_generation may be None, so use some obvious max number
        max_stagnation_length = parameters.stopping_after_n_generation or requirements.num_of_generations
        self.stop_optimisation = \
            GroupedCondition(self.log) \
            .add_condition(
                lambda: self.timer.is_time_limit_reached(self.generations.generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.generations.generation_num >= requirements.num_of_generations,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_duration >= max_stagnation_length,
                'Optimisation finished: Early stopping criteria was satisfied'
            )

    def _init_population(self, pop_size: int, max_depth: int) -> PopulationT:
        builder = InitialPopulationBuilder(self.graph_generation_params)
        if not self.initial_graph:
            random_graph_sampler = partial(random_graph, self.graph_generation_params, self.requirements, max_depth)
            builder.with_custom_sampler(random_graph_sampler)
        else:
            initial_req = deepcopy(self.requirements)
            initial_req.mutation_prob = 1

            def mutate_operator(ind: Individual):
                return self._mutate(ind, max_depth, custom_requirements=initial_req)

            initial_graphs = [self.graph_generation_params.adapter.adapt(g) for g in self.initial_graph]
            builder.with_initial_individuals(initial_graphs).with_mutated_inds(mutate_operator)

        return builder.build(pop_size)

    def _get_evaluator(self, objective_evaluator: ObjectiveEvaluate) -> EvaluationDispatcher:
        return EvaluationDispatcher(objective_evaluator,
                                    graph_adapter=self.graph_generation_params.adapter,
                                    timer=self.timer,
                                    n_jobs=self.requirements.n_jobs,
                                    collect_intermediate_metrics=self.requirements.collect_intermediate_metric,
                                    log=self.log)

    def _next_population(self, next_population: PopulationT):
        self.generations.append(next_population)
        self.optimisation_callback(next_population, self.generations)
        clean_operators_history(next_population)
        self.population = next_population
        self._operators_prob_update()

        self.log.info(f'Generation num: {self.generations.generation_num}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        self.log.info(f'no improvements for {self.generations.stagnation_duration} iterations')
        self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _operators_prob_update(self):
        """Extension point of the algorithm to adaptively modify parameters on each iteration."""
        pass

    def optimise(self, objective_evaluator: ObjectiveEvaluate,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:

        evaluator = self._get_evaluator(objective_evaluator)

        with self.timer, tqdm(total=self.requirements.num_of_generations,
                              desc='Generations', unit='gen', initial=1,
                              disable=not show_progress or self.log.verbosity_level == -1):

            pop_size = self._pop_size.initial
            self._next_population(evaluator(self._init_population(pop_size, self._graph_depth.initial)))

            while not self.stop_optimisation():
                pop_size = self._pop_size.next(self.population)
                self.max_depth = self._graph_depth.next()
                self.log.info(f'Next population size: {pop_size}; max graph depth: {self.max_depth}')

                individuals_to_select = \
                    regularized_population(self.parameters.regularization_type,
                                           self.population,
                                           evaluator,
                                           self.graph_generation_params)

                # TODO: collapse this selection & reprodue for 1 and for many
                if len(self.population) == 1:
                    new_population = self.population
                else:
                    num_of_parents = num_of_parents_in_crossover(pop_size)
                    selected_individuals = selection(types=self.parameters.selection_types,
                                                     population=individuals_to_select,
                                                     pop_size=num_of_parents,
                                                     params=self.graph_generation_params)
                    new_population = self._reproduce(selected_individuals)

                new_population = list(map(self._mutate, new_population))

                new_population = evaluator(new_population)

                with_elitism = self.with_elitism(pop_size)
                num_of_new_individuals = pop_size
                if with_elitism:
                    num_of_new_individuals -= len(self.generations.best_individuals)

                new_population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                             self.population,
                                             new_population, num_of_new_individuals,
                                             graph_params=self.graph_generation_params)

                # Add best individuals from the previous generation
                if with_elitism:
                    new_population.extend(self.generations.best_individuals)

                # Then update generation
                self._next_population(new_population)

        best = self.generations.best_individuals
        return self.to_outputs(best)

    def to_outputs(self, individuals: Iterable[Individual]) -> Union[OptGraph, List[OptGraph]]:
        graphs = [ind.graph for ind in individuals]
        # for single objective with single result return it directly
        if not self.parameters.multi_objective and len(graphs) == 1:
            return graphs[0]
        return graphs

    def with_elitism(self, pop_size: int) -> bool:
        if self.parameters.multi_objective:
            return False
        else:
            return pop_size >= self._min_population_size_with_elitism

    def _mutate(self, ind: Individual,
                max_depth: Optional[int] = None,
                custom_requirements: Optional[PipelineComposerRequirements] = None) -> Individual:
        max_depth = max_depth or self.max_depth
        requirements = custom_requirements or self.requirements
        return mutation(types=self.parameters.mutation_types,
                        params=self.graph_generation_params,
                        ind=ind, requirements=requirements,
                        max_depth=max_depth, log=self.log)

    def _reproduce(self, selected_individuals: PopulationT) -> PopulationT:
        new_population = []
        for ind_1, ind_2 in zip_longest(selected_individuals[::2], selected_individuals[1::2]):
            new_population += self._crossover_pair(ind_1, ind_2)
        return list(filter(None, new_population))

    def _crossover_pair(self, individual1: Individual, individual2: Individual) -> Sequence[Individual]:
        return crossover(self.parameters.crossover_types,
                         individual1, individual2,
                         crossover_prob=self.requirements.crossover_prob,
                         max_depth=self.max_depth, log=self.log,
                         params=self.graph_generation_params)
