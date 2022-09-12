from copy import deepcopy
from random import choice
from typing import Sequence, Callable

from fedot.core.constants import MAXIMAL_ATTEMPTS_NUMBER, EVALUATION_ATTEMPTS_NUMBER
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import Crossover
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism
from fedot.core.optimisers.gp_comp.operators.inheritance import Inheritance
from fedot.core.optimisers.gp_comp.operators.mutation import Mutation
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.operators.regularization import Regularization
from fedot.core.optimisers.gp_comp.operators.selection import Selection
from fedot.core.optimisers.gp_comp.parameters.graph_depth import AdaptiveGraphDepth
from fedot.core.optimisers.gp_comp.parameters.operators_prob import init_adaptive_operators_prob
from fedot.core.optimisers.gp_comp.parameters.population_size import init_adaptive_pop_size, PopulationSize
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.populational_optimizer import PopulationalOptimizer, EvaluationAttemptsError


class EvoGraphOptimizer(PopulationalOptimizer):
    """
    Multi-objective evolutionary graph optimizer named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: GPGraphOptimizerParameters):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, parameters)
        # Define genetic operators
        self.regularization = Regularization(parameters, graph_generation_params)
        self.selection = Selection(parameters)
        self.crossover = Crossover(parameters, requirements, graph_generation_params)
        self.mutation = Mutation(parameters, requirements, graph_generation_params)
        self.inheritance = Inheritance(parameters, self.selection)
        self.elitism = Elitism(parameters, objective.is_multi_objective)
        self.operators = [self.regularization, self.selection, self.crossover,
                          self.mutation, self.inheritance, self.elitism]

        # Define adaptive parameters
        self._pop_size: PopulationSize = init_adaptive_pop_size(parameters, self.generations)
        self._operators_prob = init_adaptive_operators_prob(parameters)
        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=requirements.start_depth,
                                               max_depth=requirements.max_depth,
                                               max_stagnation_gens=parameters.adaptive_depth_max_stagnation,
                                               adaptive=parameters.adaptive_depth)

        # Define initial parameters
        self.requirements.max_depth = self._graph_depth.initial
        self.parameters.pop_size = self._pop_size.initial
        self.initial_individuals = \
            [Individual(self.graph_generation_params.adapter.adapt(graph)) for graph in initial_graphs]

    def _initial_population(self, evaluator: Callable):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._update_population(evaluator(self.initial_individuals))

        if len(self.initial_individuals) < self.parameters.pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals)
            # Adding of extended population to history
            self._update_population(evaluator(self.initial_individuals))

    def _extend_population(self, initial_individuals) -> PopulationT:
        iter_num = 0
        initial_graphs = [ind.graph for ind in initial_individuals]
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        self.mutation.update_requirements(requirements=initial_req)
        while len(initial_individuals) < self.parameters.pop_size:
            new_ind = self.mutation(choice(self.initial_individuals))
            new_graph = new_ind.graph
            iter_num += 1
            if new_graph not in initial_graphs and self.graph_generation_params.verifier(new_graph):
                initial_individuals.append(new_ind)
                initial_graphs.append(new_graph)
            if iter_num > MAXIMAL_ATTEMPTS_NUMBER:
                self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping.'
                                 f'Current size {len(self.initial_individuals)} '
                                 f'instead of {self.parameters.pop_size} graphs.')
                break
        self.mutation.update_requirements(requirements=self.requirements)
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
            self.parameters.mutation_prob, self.parameters.crossover_prob = \
                self._operators_prob.next(self.population)
        self.parameters.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.parameters.pop_size}; '
            f'max graph depth: {self.requirements.max_depth}')

        # update requirements in operators
        for operator in self.operators:
            operator.update_requirements(self.parameters, self.requirements)

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
