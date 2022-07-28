from copy import deepcopy
from random import choice
from typing import Any, List, Optional, Sequence, Union, Callable

from fedot.core.optimisers.gp_comp.parameters.population_size import init_adaptive_pop_size, PopulationSize

from fedot.core.optimisers.gp_comp.parameters.operators_prob import init_adaptive_operators_prob

from fedot.core.optimisers.gp_comp.parameters.graph_depth import AdaptiveGraphDepth

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.constants import MAXIMAL_ATTEMPTS_NUMBER, EVALUATION_ATTEMPTS_NUMBER
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, Mutation
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, crossover_parents_selection, selection
from fedot.core.optimisers.populational_optimiser import PopulationalOptimiser
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiserParameters
from fedot.core.pipelines.pipeline import Pipeline


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
                 elitism_type: ElitismTypesEnum = ElitismTypesEnum.keep_n_best,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.elitism_type = elitism_type

        self.set_default_params()  # always initialize in proper state


class EvoGraphOptimiser(PopulationalOptimiser):
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
        self.mutation = Mutation(parameters.mutation_types, graph_generation_params, requirements)
        self.elitism = Elitism(self.parameters.elitism_type, requirements, objective.is_multi_objective)

        # Define adaptive parameters
        self._pop_size: PopulationSize = \
            init_adaptive_pop_size(parameters.genetic_scheme_type, requirements, self.generations)

        self._operators_prob = \
            init_adaptive_operators_prob(parameters.genetic_scheme_type, requirements)

        start_depth = requirements.start_depth or requirements.max_depth

        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=start_depth,
                                               max_depth=requirements.max_depth,
                                               max_stagnated_generations=parameters.depth_increase_step,
                                               adaptive=parameters.with_auto_depth_configuration)

        # Define parameters
        self.requirements.max_depth = self._graph_depth.initial

        self.requirements.pop_size = self._pop_size.initial

        self.initial_individuals = \
            [Individual(self.graph_generation_params.adapter.adapt(graph)) for graph in initial_graphs]

    def _extend_population(self, initial_individuals) -> PopulationT:
        iter_num = 0
        initial_graphs = [ind.graph for ind in initial_individuals]
        while len(initial_individuals) < self.requirements.pop_size:
            initial_req = deepcopy(self.requirements)
            initial_req.mutation_prob = 1
            self.mutation.update_requirements(initial_req)
            new_ind = self.mutation(choice(self.initial_individuals))
            new_graph = new_ind.graph
            iter_num += 1
            if new_graph not in initial_graphs and self.graph_generation_params.verifier(new_graph):
                initial_individuals.append(new_ind)
                initial_graphs.append(new_graph)
            if iter_num > MAXIMAL_ATTEMPTS_NUMBER:
                self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping.'
                                 f'Current size {len(self.initial_individuals)} '
                                 f'instead of {self.requirements.pop_size} graphs.')
                break
        self.mutation.update_requirements(self.requirements)
        return initial_individuals

    def _update_requirements(self):
        if not self.generations.is_any_improved:
            self.requirements.mutation_prob, self.requirements.crossover_prob = \
                self._operators_prob.next(self.population)
        self.requirements.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.requirements.pop_size}; max graph depth: {self.requirements.max_depth}')
        self.mutation.update_requirements(self.requirements)

    def _initial_population(self, evaluator: Callable):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._next_population(evaluator(self.initial_individuals))

        if len(self.initial_individuals) < self.requirements.pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals)
            # Adding of extended population to history
            self._next_population(evaluator(self.initial_individuals))

    def _evolve_population(self, evaluator: Callable) -> PopulationT:
        """ Method realizing full evolution cycle """
        self._update_requirements()
        individuals_to_select = regularized_population(self.parameters.regularization_type,
                                                       self.population,
                                                       evaluator,
                                                       self.graph_generation_params)

        selected_individuals = selection(types=self.parameters.selection_types,
                                         population=individuals_to_select,
                                         pop_size=self.requirements.pop_size,
                                         params=self.graph_generation_params)

        new_population = self._spawn_evaluated_population(selected_individuals=selected_individuals,
                                                          evaluator=evaluator)

        new_population = self._inheritance(new_population)

        new_population = self.elitism(self.generations.best_individuals, new_population)

        return new_population

    def _spawn_evaluated_population(self, selected_individuals: List[Individual], evaluator: Callable) -> PopulationT:
        """ Reproduce and evaluate new population. If at least one of received individuals can not be evaluated then
        mutate and evaluate selected individuals until a new population is obtained
        or the number of attempts is exceeded """

        iter_num = 0
        new_population = None
        while not new_population:
            new_population = self._reproduce(selected_individuals)
            new_population = self.mutation(new_population)
            new_population = evaluator(new_population)

            if iter_num > EVALUATION_ATTEMPTS_NUMBER:
                break
            iter_num += 1

        if not new_population:
            raise AttributeError('Too many fitness evaluation errors. Composing stopped.')

        return new_population

    def _update_requirements(self):
        if not self.generations.is_any_improved:
            self.requirements.mutation_prob, self.requirements.crossover_prob = \
                self._operators_prob.next(self.population)
        self.requirements.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.requirements.pop_size}; max graph depth: {self.requirements.max_depth}')
        self.mutation.update_requirements(self.requirements)

    def _inheritance(self, offspring: PopulationT) -> PopulationT:
        """Gather next population given new offspring, previous population and elite individuals.
        :param offspring: offspring of current population.
        :return: next population."""

        # TODO: from inheritance together with elitism we can get duplicate inds!
        offspring = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                self.population,
                                offspring, self.requirements.pop_size,
                                graph_params=self.graph_generation_params)
        return offspring

    def _reproduce(self, population: PopulationT) -> PopulationT:
        if len(population) == 1:
            new_population = population
        else:
            new_population = []
            for ind_1, ind_2 in crossover_parents_selection(population):
                new_population += self._crossover_pair(ind_1, ind_2)
        return new_population

    def _crossover_pair(self, individual1: Individual, individual2: Individual) -> PopulationT:
        return crossover(self.parameters.crossover_types, individual1, individual2,
                         max_depth=self.requirements.max_depth, crossover_prob=self.requirements.crossover_prob,
                         params=self.graph_generation_params)
