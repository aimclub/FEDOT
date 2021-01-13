import math
from copy import deepcopy
from functools import partial
from typing import (Any, Callable, List, Optional, Tuple)

import numpy as np

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.composing_history import ComposingHistory
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum, crossover
from fedot.core.composer.optimisers.gp_operators import random_chain, num_of_parents_in_crossover
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, mutation
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.composer.optimisers.selection import SelectionTypesEnum, selection
from fedot.core.composer.timer import CompositionTimer
from fedot.core.log import default_log, Log


class GPChainOptimiserParameters:
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
        :param start_depth: start value of tree depth. Using when with_auto_depth_configuration is True
    """

    def __init__(self, selection_types: List[SelectionTypesEnum] = None,
                 crossover_types: List[CrossoverTypesEnum] = None,
                 mutation_types: List[MutationTypesEnum] = None,
                 regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none,
                 genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational,
                 with_auto_depth_configuration: bool = False, depth_increase_step: int = 3,
                 start_depth: int = 3):

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.with_auto_depth_configuration = with_auto_depth_configuration
        self.depth_increase_step = depth_increase_step
        self.start_depth = start_depth
        self.set_default_params()

    def set_default_params(self):
        if not self.selection_types:
            self.selection_types = [SelectionTypesEnum.tournament]
        if not self.crossover_types:
            self.crossover_types = [CrossoverTypesEnum.subtree]
        if not self.mutation_types:
            self.mutation_types = [MutationTypesEnum.simple]


class GPChainOptimiser:
    """
    Base class of evolutionary chain optimiser

    :param initial_chain: chain which was initialized outside the optimiser
    :param requirements: composer requirements
    :param chain_generation_params: parameters for new chain generation
    :param parameters: parameters of chain optimiser
    :param log: optional parameter for log oject
    """

    def __init__(self, initial_chain, requirements, chain_generation_params,
                 parameters: Optional[GPChainOptimiserParameters] = None, log: Log = None):
        self.chain_generation_params = chain_generation_params
        self.primary_node_func = self.chain_generation_params.primary_node_func
        self.secondary_node_func = self.chain_generation_params.secondary_node_func
        self.chain_class = self.chain_generation_params.chain_class
        self.requirements = requirements
        self.parameters = GPChainOptimiserParameters() if parameters is None else parameters
        self.max_depth = self.parameters.start_depth if self.parameters.with_auto_depth_configuration else \
            self.requirements.max_depth

        self.generation_num = 0
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self.chain_generation_function = partial(random_chain, chain_generation_params=self.chain_generation_params,
                                                 requirements=self.requirements, max_depth=self.max_depth)

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            ex = f'Object chain_class has no required attributes for gp_optimizer'
            self.log.error(ex)
            raise AttributeError(ex)

        if not self.requirements.pop_size:
            self.requirements.pop_size = 10

        if initial_chain and type(initial_chain) != list:
            self.population = [deepcopy(initial_chain) for _ in range(self.requirements.pop_size)]
        else:
            self.population = initial_chain

        self.history = ComposingHistory()

    def optimise(self, objective_function, offspring_rate: float = 0.5,
                 on_next_iteration_callback: Optional[Callable] = None):
        if on_next_iteration_callback is None:
            on_next_iteration_callback = self.default_on_next_iteration_callback

        if self.population is None:
            self.population = self._make_population(self.requirements.pop_size)

        num_of_new_individuals = self.offspring_size(offspring_rate)

        with CompositionTimer() as t:

            if self.requirements.add_single_model_chains:
                best_single_model, self.requirements.primary = \
                    self._best_single_models(objective_function)

            for ind in self.population:
                ind.fitness = objective_function(ind)

            on_next_iteration_callback(self.population)

            self.log.info(f'Best metric is {self.best_individual.fitness}')

            for self.generation_num in range(self.requirements.num_of_generations - 1):
                self.log.info(f'Generation num: {self.generation_num}')
                self.num_of_gens_without_improvements = self.update_stagnation_counter()
                self.log.info(
                    f'max_depth: {self.max_depth}, no improvements: {self.num_of_gens_without_improvements}')

                if self.parameters.with_auto_depth_configuration and self.generation_num != 0:
                    self.max_depth_recount()

                individuals_to_select = regularized_population(reg_type=self.parameters.regularization_type,
                                                               population=self.population,
                                                               objective_function=objective_function,
                                                               chain_class=self.chain_class)

                num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_parents)

                new_population = []

                for parent_num in range(0, len(selected_individuals), 2):
                    new_population += self.reproduce(selected_individuals[parent_num],
                                                     selected_individuals[parent_num + 1])

                    new_population[parent_num].fitness = objective_function(new_population[parent_num])
                    new_population[parent_num + 1].fitness = objective_function(new_population[parent_num + 1])

                self.prev_best = deepcopy(self.best_individual)

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop)

                if self.with_elitism:
                    self.population.append(self.prev_best)

                on_next_iteration_callback(self.population)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log.info(f'Best metric is {self.best_individual.fitness}')

                if t.is_time_limit_reached(self.requirements.max_lead_time, self.generation_num):
                    break

            best = self.best_individual

            if self.requirements.add_single_model_chains and \
                    (best_single_model.fitness <= best.fitness):
                best = best_single_model
        return best

    @property
    def best_individual(self) -> Any:
        return self.get_best_individual(self.population)

    @property
    def with_elitism(self) -> bool:
        return self.requirements.pop_size > 1

    @property
    def num_of_inds_in_next_pop(self):
        return self.requirements.pop_size - 1 if self.with_elitism else self.requirements.pop_size

    def update_stagnation_counter(self) -> int:
        value = 0
        if self.generation_num != 0:
            if self.is_equal_fitness(self.prev_best.fitness, self.best_individual.fitness):
                value = self.num_of_gens_without_improvements + 1

        return value

    def max_depth_recount(self):
        if self.num_of_gens_without_improvements == self.parameters.depth_increase_step and \
                self.max_depth + 1 <= self.requirements.max_depth:
            self.max_depth += 1

    def get_best_individual(self, individuals: List[Any], equivalents_from_current_pop=True) -> Any:
        best_ind = min(individuals, key=lambda ind: ind.fitness)
        if equivalents_from_current_pop:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind)
        else:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind, individuals)

        if equivalents:
            best_candidate_id = min(equivalents, key=equivalents.get)
            best_ind = individuals[best_candidate_id]
        return best_ind

    def simpler_equivalents_of_best_ind(self, best_ind: Any, inds: List[Any] = None) -> dict:
        individuals = self.population if inds is None else inds

        sort_inds = np.argsort([ind.fitness for ind in individuals])[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = self.is_equal_fitness(best_ind.fitness, individuals[i].fitness)
            has_less_num_of_models_than_best = len(individuals[i].nodes) < len(best_ind.nodes)
            if is_fitness_equals_to_best and has_less_num_of_models_than_best:
                simpler_equivalents[i] = len(individuals[i].nodes)
        return simpler_equivalents

    def reproduce(self, selected_individual_first, selected_individual_second=None) -> Tuple[Any]:
        if selected_individual_second:
            new_inds = crossover(self.parameters.crossover_types,
                                 selected_individual_first,
                                 selected_individual_second,
                                 crossover_prob=self.requirements.crossover_prob,
                                 max_depth=self.requirements.max_depth)
        else:
            new_inds = [selected_individual_first]

        new_inds = tuple([mutation(types=self.parameters.mutation_types,
                                   chain_generation_params=self.chain_generation_params,
                                   chain=new_ind, requirements=self.requirements,
                                   max_depth=self.max_depth) for new_ind in new_inds])

        return new_inds

    def _make_population(self, pop_size: int) -> List[Any]:
        model_chains = []
        while len(model_chains) < pop_size:
            chain = self.chain_generation_function()
            if constraint_function(chain):
                model_chains.append(chain)
        return model_chains

    def _best_single_models(self, objective_function: Callable, num_best: int = 7):
        single_models_inds = []
        for model in self.requirements.primary:
            single_models_ind = self.chain_class([self.primary_node_func(model)])
            single_models_ind.fitness = objective_function(single_models_ind)
            single_models_inds.append(single_models_ind)
        best_inds = sorted(single_models_inds, key=lambda ind: ind.fitness)
        return best_inds[0], [i.nodes[0].model.model_type for i in best_inds][:num_best]

    def offspring_size(self, offspring_rate: float = None):
        default_offspring_rate = 0.5 if not offspring_rate else offspring_rate
        if self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
            num_of_new_individuals = math.ceil(self.requirements.pop_size * default_offspring_rate)
        else:
            num_of_new_individuals = self.requirements.pop_size - 1
        return num_of_new_individuals

    def is_equal_fitness(self, first_fitness, second_fitness, atol=1e-10, rtol=1e-10):
        return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)

    def default_on_next_iteration_callback(self, individuals):
        self.history.add_to_history(individuals)