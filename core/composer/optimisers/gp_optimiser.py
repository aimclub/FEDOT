from copy import deepcopy
from dataclasses import dataclass
from random import choice, randint
from typing import (
    List,
    Callable,
    Any,
    Optional
)

import numpy as np
from core.composer.optimisers.regularization import RegularizationTypeEnum
from core.composer.optimisers.crossover import standard_crossover
from core.composer.optimisers.gp_operators import node_height
from core.composer.optimisers.mutation import standard_mutation
from core.composer.optimisers.selection import tournament_selection
from core.composer.optimisers.regularization import regularized_population
from core.composer.timer import CompositionTimer


@dataclass
class GPChainOptimiserParameters:
    selection_type: Optional[Callable] = tournament_selection
    crossover_type: Optional[Callable] = standard_crossover
    mutation_type: Optional[Callable] = standard_mutation
    regularization_type: RegularizationTypeEnum = RegularizationTypeEnum.decremental


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

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            raise AttributeError(f'Object chain_base has no required attributes for gp_optimizer')

        if initial_chain and type(initial_chain) != list:
            self.population = [initial_chain] * requirements.pop_size
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def optimise(self, metric_function_for_nodes, ):

        with CompositionTimer() as t:

            history = []

            self.fitness = [round(metric_function_for_nodes(chain), 3) for chain in self.population]

            [history.append((self.population[ind_num], self.fitness[ind_num])) for ind_num in
             range(self.requirements.pop_size)]

            for generation_num in range(self.requirements.num_of_generations - 1):
                print(f'GP generation num: {generation_num}')
                best_ind_num = np.argmin(self.fitness)
                self.best_individual = deepcopy(self.population[best_ind_num])
                self.best_fitness = self.fitness[best_ind_num]

                individuals_to_select = self.population + regularized_population(self.parameters.regularization_type,
                                                                                 self.population, self.requirements,
                                                                                 metric_function_for_nodes,
                                                                                 self.chain_class)

                selected_individuals = self.parameters.selection_type(self.fitness, individuals_to_select)

                for ind_num in range(self.requirements.pop_size):

                    if ind_num == self.requirements.pop_size - 1:
                        self.population[ind_num] = deepcopy(self.best_individual)
                        self.fitness[ind_num] = self.best_fitness
                        history.append((self.population[ind_num], self.fitness[ind_num]))
                        break

                    self.population[ind_num] = self.parameters.crossover_type(*selected_individuals[ind_num],
                                                                  crossover_prob=self.requirements.crossover_prob,
                                                                  max_depth=self.requirements.max_depth)

                    self.population[ind_num] = self.parameters.mutation_type(chain=self.population[ind_num],
                                                                 secondary=self.requirements.secondary,
                                                                 primary=self.requirements.primary,
                                                                 secondary_node_func=self.secondary_node_func,
                                                                 primary_node_func=self.primary_node_func,
                                                                 mutation_prob=self.requirements.mutation_prob)

                    self.fitness[ind_num] = round(metric_function_for_nodes(self.population[ind_num]), 3)
                    print(f'Best metric is {np.min(self.fitness)}')

                    history.append((self.population[ind_num], self.fitness[ind_num]))

                if t.is_max_time_reached(self.requirements.max_lead_time, generation_num):
                    break

        return self.population[np.argmin(self.fitness)], history

    def _make_population(self, pop_size: int) -> List[Any]:
        return [self._random_chain() for _ in range(pop_size)]

    def _random_chain(self) -> Any:
        chain = self.chain_class()
        chain_root = self.secondary_node_func(model_type=choice(self.requirements.secondary))
        chain.add_node(chain_root)
        self._chain_growth(chain, chain_root)
        return chain

    def _chain_growth(self, chain: Any, node_parent: Any):
        offspring_size = randint(2, self.requirements.max_arity)
        for offspring_node in range(offspring_size):
            height = node_height(chain, node_parent)
            is_max_depth_exceeded = height >= self.requirements.max_depth - 1
            is_primary_node_selected = height < self.requirements.max_depth - 1 and randint(0, 1)
            if is_max_depth_exceeded or is_primary_node_selected:
                primary_node = self.primary_node_func(model_type=choice(self.requirements.primary))
                node_parent.nodes_from.append(primary_node)
                chain.add_node(primary_node)
            else:
                secondary_node = self.secondary_node_func(model_type=choice(self.requirements.secondary))
                chain.add_node(secondary_node)
                node_parent.nodes_from.append(secondary_node)
                self._chain_growth(chain, secondary_node)
