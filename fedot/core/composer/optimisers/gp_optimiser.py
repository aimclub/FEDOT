import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, List, Optional, Tuple)

import numpy as np

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum, crossover
from fedot.core.composer.optimisers.gp_operators import random_chain
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.composer.optimisers.mutation import MutationTypesEnum, mutation
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.composer.optimisers.selection import SelectionTypesEnum, selection
from fedot.core.composer.timer import CompositionTimer
from fedot.core.log import default_log


@dataclass
class GPChainOptimiserParameters:
    def __init__(self, selection_types: List[SelectionTypesEnum] = None,
                 crossover_types: List[CrossoverTypesEnum] = None,
                 mutation_types: List[MutationTypesEnum] = None,
                 regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.decremental,
                 genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.steady_state):

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.set_default_params()

    def set_default_params(self):
        if not self.selection_types:
            self.selection_types = [SelectionTypesEnum.tournament]
        if not self.crossover_types:
            self.crossover_types = [CrossoverTypesEnum.subtree]
        if not self.mutation_types:
            self.mutation_types = [MutationTypesEnum.simple]


class GPChainOptimiser:
    def __init__(self, initial_chain, requirements, primary_node_func: Callable, secondary_node_func: Callable,
                 chain_class: Callable, parameters: Optional[GPChainOptimiserParameters] = None,
                 log=default_log(__name__)):
        self.requirements = requirements
        self.primary_node_func = primary_node_func
        self.secondary_node_func = secondary_node_func
        self.history = []
        self.chain_class = chain_class
        self.parameters = GPChainOptimiserParameters() if parameters is None else parameters
        self.chain_generation_function = partial(random_chain, chain_class=chain_class, requirements=self.requirements,
                                                 primary_node_func=self.primary_node_func,
                                                 secondary_node_func=self.secondary_node_func)
        self.log = log

        necessary_attrs = ['add_node', 'root_node', 'replace_node_with_parents', 'update_node', 'node_childs']
        if not all([hasattr(self.chain_class, attr) for attr in necessary_attrs]):
            ex = f'Object chain_class has no required attributes for gp_optimizer'
            self.log.error(ex)
            raise AttributeError(ex)

        if initial_chain and type(initial_chain) != list:
            self.population = [deepcopy(initial_chain) for _ in range(requirements.pop_size)]
        else:
            self.population = initial_chain or self._make_population(self.requirements.pop_size)

    def optimise(self, objective_function, offspring_rate=0.5):

        if self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
            num_of_new_individuals = math.ceil(self.requirements.pop_size * offspring_rate)
        else:
            num_of_new_individuals = self.requirements.pop_size - 1

        with CompositionTimer() as t:

            self.history = []

            if self.requirements.add_single_model_chains:
                best_single_model, self.requirements.primary = \
                    self._best_single_models(objective_function)
                self.history.append(best_single_model)

            for ind in self.population:
                ind.fitness = objective_function(ind)

            self._add_to_history(self.population)

            for generation_num in range(self.requirements.num_of_generations - 1):
                print(f'Generation num: {generation_num}')

                individuals_to_select = regularized_population(reg_type=self.parameters.regularization_type,
                                                               population=self.population,
                                                               objective_function=objective_function,
                                                               chain_class=self.chain_class)

                num_of_parents = num_of_new_individuals if not num_of_new_individuals % 2 else num_of_new_individuals + 1
                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_parents)

                new_population = []

                for parent_num in range(0, len(selected_individuals), 2):
                    new_population += self.reproduce(selected_individuals[parent_num],
                                                     selected_individuals[parent_num + 1])

                    new_population[parent_num].fitness = objective_function(new_population[parent_num])
                    new_population[parent_num + 1].fitness = objective_function(new_population[parent_num + 1])

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.requirements.pop_size - 1)

                self.population.append(self.best_individual)

                self._add_to_history(self.population)

                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log.info(f'Best metric is {self.best_individual.fitness}')

                if t.is_time_limit_reached(self.requirements.max_lead_time, generation_num):
                    break
            best = self.best_individual
            if self.requirements.add_single_model_chains and \
                    (best_single_model.fitness <= best.fitness):
                best = best_single_model
        return best, self.history

    @property
    def best_individual(self) -> Any:
        best_ind = min(self.population, key=lambda ind: ind.fitness)
        equivalents = self.simpler_equivalents_of_best_ind(best_ind)

        if equivalents:
            best_candidate_id = min(equivalents, key=equivalents.get)
            best_ind = self.population[best_candidate_id]
        return best_ind

    def simpler_equivalents_of_best_ind(self, best_ind: Any) -> dict:
        sort_inds = np.argsort([ind.fitness for ind in self.population])[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = best_ind.fitness == self.population[i].fitness
            has_less_num_of_models_than_best = len(self.population[i].nodes) < len(best_ind.nodes)
            if is_fitness_equals_to_best and has_less_num_of_models_than_best:
                simpler_equivalents[i] = len(self.population[i].nodes)
        return simpler_equivalents

    def reproduce(self, selected_individual_first, selected_individual_second) -> Tuple[Any]:
        new_inds = crossover(self.parameters.crossover_types,
                             selected_individual_first,
                             selected_individual_second,
                             crossover_prob=self.requirements.crossover_prob,
                             max_depth=self.requirements.max_depth)

        new_inds = tuple([mutation(types=self.parameters.mutation_types,
                                   chain_class=self.chain_class,
                                   chain=new_ind,
                                   requirements=self.requirements,
                                   secondary_node_func=self.secondary_node_func,
                                   primary_node_func=self.primary_node_func,
                                   mutation_prob=self.requirements.mutation_prob) for new_ind in new_inds])
        return new_inds

    def _make_population(self, pop_size: int) -> List[Any]:
        model_chains = []
        while len(model_chains) < pop_size:
            chain = self.chain_generation_function()
            if constraint_function(chain):
                model_chains.append(chain)
        return model_chains

    def _add_to_history(self, individuals: List[Any]):
        [self.history.append(ind) for ind in individuals]

    def _best_single_models(self, objective_function: Callable, num_best: int = 7):
        single_models_inds = []
        for model in self.requirements.primary:
            single_models_ind = self.chain_class([self.primary_node_func(model)])
            single_models_ind.fitness = objective_function(single_models_ind)
            single_models_inds.append(single_models_ind)
        best_inds = sorted(single_models_inds, key=lambda ind: ind.fitness)
        return best_inds[0], [i.nodes[0].model.model_type for i in best_inds][:num_best]
