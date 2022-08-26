from copy import deepcopy
from dataclasses import dataclass
from random import choice
from typing import Any, List, Optional, Sequence, Union, Callable

from fedot.core.constants import MAXIMAL_ATTEMPTS_NUMBER, EVALUATION_ATTEMPTS_NUMBER
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, Crossover
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, Inheritance
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, Mutation
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, Regularization
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, Selection
from fedot.core.optimisers.gp_comp.parameters.graph_depth import AdaptiveGraphDepth
from fedot.core.optimisers.gp_comp.parameters.operators_prob import init_adaptive_operators_prob
from fedot.core.optimisers.gp_comp.parameters.population_size import init_adaptive_pop_size, PopulationSize
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizerParameters
from fedot.core.optimisers.populational_optimizer import PopulationalOptimizer, EvaluationAttemptsError
from fedot.core.pipelines.pipeline import Pipeline


@dataclass
class GPGraphOptimizerParameters(GraphOptimizerParameters):
    """
    This class is for defining the parameters of optimiser

    :param selection_types: Sequence of selection operators types
    :param crossover_types: Sequence of crossover operators types
    :param mutation_types: Sequence of mutation operators types
    :param regularization_type: type of regularization operator
    :param genetic_scheme_type: type of genetic evolutionary scheme
    :param elitism_type: type of elitism operator evolution
    """

    selection_types: Sequence[SelectionTypesEnum] = ()
    crossover_types: Sequence[Union[CrossoverTypesEnum, Any]] = \
        (CrossoverTypesEnum.subtree,
         CrossoverTypesEnum.one_point)
    mutation_types: Sequence[Union[MutationTypesEnum, Any]] = \
        (MutationTypesEnum.simple,
         MutationTypesEnum.reduce,
         MutationTypesEnum.growth,
         MutationTypesEnum.local_growth)
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none
    genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational
    elitism_type: ElitismTypesEnum = ElitismTypesEnum.keep_n_best

    def __post_init__(self):
        if not self.selection_types:
            if self.multi_objective:
                self.selection_types = (SelectionTypesEnum.spea2,)
            else:
                self.selection_types = (SelectionTypesEnum.tournament,)


class EvoGraphOptimizer(PopulationalOptimizer):
    """
    Multi-objective evolutionary graph optimizer named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimizerParameters] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, parameters)
        # Define genetic operators
        self.regularization = Regularization(parameters.regularization_type, requirements, graph_generation_params,)
        self.selection = Selection(parameters.selection_types, requirements)
        self.crossover = Crossover(parameters.crossover_types, requirements, graph_generation_params)
        self.mutation = Mutation(parameters.mutation_types, requirements, graph_generation_params)
        self.inheritance = Inheritance(parameters.genetic_scheme_type, self.selection, requirements)
        self.elitism = Elitism(parameters.elitism_type, requirements, objective.is_multi_objective)
        self.operators = [self.regularization, self.selection, self.crossover,
                          self.mutation, self.inheritance, self.elitism]

        # Define adaptive parameters
        self._pop_size: PopulationSize = \
            init_adaptive_pop_size(parameters.genetic_scheme_type, requirements, self.generations)
        self._operators_prob = \
            init_adaptive_operators_prob(parameters.genetic_scheme_type, requirements)
        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=requirements.start_depth,
                                               max_depth=requirements.max_depth,
                                               stagnation_threshold=requirements.adaptive_depth_max_stagnation,
                                               adaptive=requirements.adaptive_depth)

        # Define initial parameters
        self.requirements.max_depth = self._graph_depth.initial
        self.requirements.pop_size = self._pop_size.initial
        self.initial_individuals = \
            [Individual(self.graph_generation_params.adapter.adapt(graph)) for graph in initial_graphs]

    def _initial_population(self, evaluator: Callable):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._update_population(evaluator(self.initial_individuals))

        if len(self.initial_individuals) < self.requirements.pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals)
            # Adding of extended population to history
            self._update_population(evaluator(self.initial_individuals))

    def _extend_population(self, initial_individuals) -> PopulationT:
        iter_num = 0
        initial_graphs = [ind.graph for ind in initial_individuals]
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        self.mutation.update_requirements(initial_req)
        while len(initial_individuals) < self.requirements.pop_size:
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

    def _evolve_population(self, evaluator: Callable) -> PopulationT:
        """ Method realizing full evolution cycle """
        self._update_requirements()

        individuals_to_select = self.regularization(self.population, evaluator)
        selected_individuals = self.selection(individuals_to_select)
        new_population = self._spawn_evaluated_population(selected_individuals=selected_individuals,
                                                          evaluator=evaluator)
        new_population = self.inheritance(self.population, new_population)
        new_population = self.elitism(self.generations.best_individuals, new_population)

        return new_population

    def _update_requirements(self):
        if not self.generations.is_any_improved:
            self.requirements.mutation_prob, self.requirements.crossover_prob = \
                self._operators_prob.next(self.population)
        self.requirements.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.requirements.pop_size}; max graph depth: {self.requirements.max_depth}')
        self._update_evolutionary_operators_requirements(self.requirements)

    def _update_evolutionary_operators_requirements(self, new_requirements: PipelineComposerRequirements):
        operators_list = self.operators
        for operator in operators_list:
            operator.update_requirements(new_requirements)

    def _spawn_evaluated_population(self, selected_individuals: PopulationT, evaluator: Callable) -> PopulationT:
        """ Reproduce and evaluate new population. If at least one of received individuals can not be evaluated then
        mutate and evaluate selected individuals until a new population is obtained
        or the number of attempts is exceeded """

        iter_num = 0
        new_population = None
        while not new_population:
            new_population = self.crossover(selected_individuals)
            new_population = self.mutation(new_population)
            new_population = evaluator(new_population)

            if iter_num > EVALUATION_ATTEMPTS_NUMBER:
                break
            iter_num += 1

        if not new_population:
            raise EvaluationAttemptsError()

        return new_population
