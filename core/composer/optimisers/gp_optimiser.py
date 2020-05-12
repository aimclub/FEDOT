from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    List,
    Callable,
    Any,
    Optional,
    Tuple
)

import numpy as np

from core.composer.optimisers.crossover import CrossoverTypesEnum
from core.composer.optimisers.crossover import crossover
from core.composer.optimisers.gp_operators import random_chain
from core.composer.optimisers.mutation import MutationTypesEnum
from core.composer.optimisers.mutation import mutation
from core.composer.optimisers.regularization import RegularizationTypesEnum
from core.composer.optimisers.regularization import regularized_population
from core.composer.optimisers.selection import SelectionTypesEnum
from core.composer.optimisers.selection import selection
from core.composer.timer import CompositionTimer


@dataclass
class GPChainOptimiserParameters:
    selection_types: List[SelectionTypesEnum] = None
    crossover_types: List[CrossoverTypesEnum] = None
    mutation_types: List[MutationTypesEnum] = None
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.decremental

    def __post_init__(self):
        if not self.selection_types:
            self.selection_types = [SelectionTypesEnum.tournament]
        if not self.crossover_types:
            self.crossover_types = [CrossoverTypesEnum.subtree]
        if not self.mutation_types:
            self.mutation_types = [MutationTypesEnum.simple]


class GPChainOptimiser:
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable,
                 chain_class: Callable, parameters: Optional[GPChainOptimiserParameters] = None):
        self.requirements = requirements
        self.primary_node_func = primary_node_func
        self.secondary_node_func = secondary_node_func
        self.best_individual = None
        self.best_fitness = None
        self.chain_class = chain_class
        self.parameters = GPChainOptimiserParameters() if parameters is None else parameters
        self.chain_generation_function = partial(random_chain, chain_class=chain_class, requirements=self.requirements,
                                                 primary_node_func=self.primary_node_func,
                                                 secondary_node_func=self.secondary_node_func)

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            raise AttributeError(f'Object chain_class has no required attributes for gp_optimizer')

        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def optimise(self, metric_function_for_nodes):

        with CompositionTimer() as t:

            history = []

            self.fitness = [metric_function_for_nodes(chain) for chain in self.population]

            [history.append((self.population[ind_num], self.fitness[ind_num])) for ind_num in
             range(self.requirements.pop_size)]

            for generation_num in range(self.requirements.num_of_generations - 1):
                print(f'GP generation num: {generation_num}')

                self.best_individual, self.best_fitness = self.best_individual_with_fitness

                additional_inds, fitness = regularized_population(self.parameters.regularization_type,
                                                                  self.population, self.requirements,
                                                                  metric_function_for_nodes,
                                                                  self.chain_class)

                individuals_to_select = self.population + additional_inds
                fitness = self.fitness + fitness

                selected_individuals = selection(self.parameters.selection_types, fitness, individuals_to_select,
                                                 self.requirements.pop_size)

                for ind_num in range(self.requirements.pop_size):

                    if ind_num == self.requirements.pop_size - 1:
                        self.population[ind_num] = deepcopy(self.best_individual)
                        self.fitness[ind_num] = self.best_fitness
                        history.append((self.population[ind_num], self.fitness[ind_num]))
                        break

                    self.population[ind_num] = crossover(self.parameters.crossover_types,
                                                         *selected_individuals[ind_num],
                                                         crossover_prob=self.requirements.crossover_prob,
                                                         max_depth=self.requirements.max_depth)

                    self.population[ind_num] = mutation(types=self.parameters.mutation_types,
                                                        chain_class=self.chain_class,
                                                        chain=self.population[ind_num],
                                                        requirements=self.requirements,
                                                        secondary_node_func=self.secondary_node_func,
                                                        primary_node_func=self.primary_node_func,
                                                        mutation_prob=self.requirements.mutation_prob)

                    self.fitness[ind_num] = metric_function_for_nodes(self.population[ind_num])
                    print(f'Best metric is {np.min(self.fitness)}')

                    history.append((self.population[ind_num], self.fitness[ind_num]))

                print("spent time:", t.minutes_from_start)
                if t.is_max_time_reached(self.requirements.max_lead_time, generation_num):
                    break
        self.best_individual, _ = self.best_individual_with_fitness
        return self.best_individual, history

    @property
    def best_individual_with_fitness(self) -> Tuple[Any, float]:
        best_ind_num = np.argmin(self.fitness)
        equivalents = self.simpler_equivalents_of_best_ind(best_ind_num)

        if equivalents:
            best_candidate = min(equivalents, key=equivalents.get)
            best = self.population[best_candidate]
            best_fitness = self.fitness[best_candidate]
        else:
            best = deepcopy(self.population[best_ind_num])
            best_fitness = self.fitness[best_ind_num]
        return best, best_fitness

    def simpler_equivalents_of_best_ind(self, best_ind_num: int) -> dict:
        sort_inds = np.argsort(self.fitness)[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = self.fitness[best_ind_num] == self.fitness[i]
            has_less_num_of_models_than_best = len(self.population[i].nodes) < len(self.population[best_ind_num].nodes)
            if is_fitness_equals_to_best and has_less_num_of_models_than_best:
                simpler_equivalents[i] = len(self.population[i].nodes)
        return simpler_equivalents

    def _make_population(self, pop_size: int) -> List[Any]:
        return [self.chain_generation_function() for _ in range(pop_size)]
