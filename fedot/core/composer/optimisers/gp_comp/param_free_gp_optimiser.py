from copy import deepcopy
from typing import (Any, List, Optional, Tuple)

import numpy as np
from deap import tools

from fedot.core.composer.iterator import SequenceIterator, fibonacci_sequence
from fedot.core.composer.optimisers.gp_comp.gp_operators import clean_operators_history, duplicates_filtration, \
    num_of_parents_in_crossover
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiser, GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.composer.optimisers.gp_comp.operators.regularization import regularized_population
from fedot.core.composer.optimisers.gp_comp.operators.selection import selection
from fedot.core.composer.optimisers.utils.population_utils import is_equal_archive
from fedot.core.composer.timer import CompositionTimer
from fedot.core.log import Log
from fedot.core.repository.quality_metrics_repository import ComplexityMetricsEnum, MetricsEnum, MetricsRepository

DEFAULT_MAX_POP_SIZE = 55


class GPChainParameterFreeOptimiser(GPChainOptimiser):
    """
    Implementation of the parameter-free adaptive evolutionary optimiser
    (population size and genetic operators rates is changing over time).
    For details, see original paper: https://arxiv.org/abs/2001.10178
    :param initial_chain: chain which was initialized outside the optimiser
    :param requirements: composer requirements
    :param chain_generation_params: parameters for new chain generation
    :param metrics: quality metrics
    :param parameters: parameters of chain optimiser
    :param max_population_size: maximum population size
    :param log: optional parameter for log object
    :param archive_type: type of archive with best individuals

    """

    def __init__(self, initial_chain, requirements, chain_generation_params, metrics: List[MetricsEnum],
                 parameters: Optional[GPChainOptimiserParameters] = None,
                 max_population_size: int = DEFAULT_MAX_POP_SIZE,
                 sequence_function=fibonacci_sequence, log: Log = None, archive_type=None):
        super().__init__(initial_chain, requirements, chain_generation_params, metrics, parameters, log, archive_type)

        if self.parameters.genetic_scheme_type != GeneticSchemeTypesEnum.parameter_free:
            self.log.warn(f'Invalid genetic scheme type was changed to parameter-free. Continue.')
            self.parameters.genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        self.sequence_function = sequence_function
        self.max_pop_size = max_population_size
        self.iterator = SequenceIterator(sequence_func=self.sequence_function, min_sequence_value=1,
                                         max_sequence_value=self.max_pop_size,
                                         start_value=self.requirements.pop_size)

        self.requirements.pop_size = self.iterator.next()
        self.metrics = metrics

        self.qual_position = 0
        self.compl_position = 1

    def optimise(self, objective_function, offspring_rate: float = 0.5, on_next_iteration_callback=None):
        if on_next_iteration_callback is None:
            on_next_iteration_callback = self.default_on_next_iteration_callback

        if self.population is None:
            self.population = self._make_population(self.requirements.pop_size)

        num_of_new_individuals = self.offspring_size(offspring_rate)
        self.log.info(f'pop size: {self.requirements.pop_size}, num of new inds: {num_of_new_individuals}')

        with CompositionTimer(max_lead_time=self.requirements.max_lead_time, log=self.log) as t:
            if self.requirements.allow_single_operations:
                self.best_single_operation, self.requirements.primary = \
                    self._best_single_operations(objective_function, timer=t)

            self._evaluate_individuals(self.population, objective_function, timer=t)

            if self.archive is not None:
                self.archive.update(self.population)

            on_next_iteration_callback(self.population, self.archive)

            self.log_info_about_best()

            while t.is_time_limit_reached(self.generation_num) is False \
                    and self.generation_num != self.requirements.num_of_generations - 1:

                self.log.info(f'Generation num: {self.generation_num}')

                self.num_of_gens_without_improvements = self.update_stagnation_counter()
                self.log.info(f'max_depth: {self.max_depth}, no improvements: {self.num_of_gens_without_improvements}')

                if self.parameters.with_auto_depth_configuration and self.generation_num != 0:
                    self.max_depth_recount()

                self.max_std = self.update_max_std()

                individuals_to_select = regularized_population(reg_type=self.parameters.regularization_type,
                                                               population=self.population,
                                                               objective_function=objective_function,
                                                               chain_class=self.chain_class, timer=t)

                if self.parameters.multi_objective:
                    filtered_archive_items = duplicates_filtration(archive=self.archive,
                                                                   population=individuals_to_select)
                    individuals_to_select = deepcopy(individuals_to_select) + filtered_archive_items

                if num_of_new_individuals == 1 and len(self.population) == 1:
                    new_population = list(self.reproduce(self.population[0]))
                    self._evaluate_individuals(new_population, objective_function, timer=t)
                else:
                    num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                    selected_individuals = selection(types=self.parameters.selection_types,
                                                     population=individuals_to_select,
                                                     pop_size=num_of_parents)

                    new_population = []

                    for parent_num in range(0, len(selected_individuals), 2):
                        new_population += self.reproduce(selected_individuals[parent_num],
                                                         selected_individuals[parent_num + 1])

                    self._evaluate_individuals(new_population, objective_function, timer=t)

                self.requirements.pop_size = self.next_population_size(new_population)
                num_of_new_individuals = self.offspring_size(offspring_rate)
                self.log.info(f'pop size: {self.requirements.pop_size}, num of new inds: {num_of_new_individuals}')

                self.prev_best = deepcopy(self.best_individual)

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop)

                if not self.parameters.multi_objective and self.with_elitism:
                    self.population.append(self.prev_best)

                if self.archive is not None:
                    self.archive.update(self.population)

                on_next_iteration_callback(self.population, self.archive)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log_info_about_best()

                self.generation_num += 1
                clean_operators_history(self.population)

            best = self.result_individual()
            self.log.info('Result:')
            self.log_info_about_best()

        output = [ind.chain for ind in best] if isinstance(best, list) else best.chain
        return output

    @property
    def with_elitism(self) -> bool:
        if self.parameters.multi_objective:
            return False
        else:
            return self.requirements.pop_size >= 7

    @property
    def current_std(self):
        if self.parameters.multi_objective:
            std = np.std([self.get_main_metric(ind) for ind in self.population])
        else:
            std = np.std([ind.fitness for ind in self.population])
        return std

    def update_max_std(self):
        if self.generation_num == 0:
            std_max = self.current_std
            if len(self.population) == 1:
                self.requirements.mutation_prob = 1
                self.requirements.crossover_prob = 0
            else:
                self.requirements.mutation_prob = 0.5
                self.requirements.crossover_prob = 0.5
        else:
            if self.max_std < self.current_std:
                std_max = self.current_std
            else:
                std_max = self.max_std
        return std_max

    def _check_mo_improvements(self, offspring: List[Any]) -> Tuple[bool, bool]:
        complexity_decreased = False
        fitness_improved = False
        offspring_archive = tools.ParetoFront()
        offspring_archive.update(offspring)
        is_archive_improved = not is_equal_archive(self.archive, offspring_archive)
        if is_archive_improved:
            best_ind_in_prev = min(self.archive.items, key=self.get_main_metric)
            best_ind_in_current = min(offspring_archive.items, key=self.get_main_metric)
            fitness_improved = self.get_main_metric(best_ind_in_current) < self.get_main_metric(best_ind_in_prev)
            for offspring_ind in offspring_archive.items:
                if self.get_main_metric(offspring_ind) <= self.get_main_metric(best_ind_in_prev) \
                        and self.get_suppl_metric(offspring_ind) < self.get_suppl_metric(best_ind_in_prev):
                    complexity_decreased = True
                    break
        return fitness_improved, complexity_decreased

    def _check_so_improvements(self, offspring: List[Any]) -> Tuple[bool, bool]:
        suppl_metric = MetricsRepository().metric_by_id(ComplexityMetricsEnum.node_num)
        best_in_offspring = self.get_best_individual(offspring, equivalents_from_current_pop=False)
        fitness_improved = best_in_offspring.fitness < self.best_individual.fitness
        complexity_decreased = suppl_metric(best_in_offspring.chain) < suppl_metric(
            self.best_individual.chain) and best_in_offspring.fitness <= self.best_individual.fitness
        return fitness_improved, complexity_decreased

    def next_population_size(self, offspring: List[Any]) -> int:
        improvements_checker = self._check_so_improvements
        if self.parameters.multi_objective:
            improvements_checker = self._check_mo_improvements
        fitness_improved, complexity_decreased = improvements_checker(offspring)
        is_max_pop_size_reached = not self.iterator.has_next()
        progress_in_both_goals = fitness_improved and complexity_decreased and not is_max_pop_size_reached
        no_progress = not fitness_improved and not complexity_decreased and not is_max_pop_size_reached
        if (progress_in_both_goals and len(self.population) > 2) or no_progress:
            if progress_in_both_goals:
                if self.iterator.has_prev():
                    next_population_size = self.iterator.prev()
                else:
                    next_population_size = len(self.population)
            else:
                next_population_size = self.iterator.next()

                self.requirements.mutation_prob, self.requirements.crossover_prob = self.operators_prob_update(
                    std=float(self.current_std), max_std=float(self.max_std))

        else:
            next_population_size = len(self.population)
        return next_population_size

    def operators_prob_update(self, std: float, max_std: float) -> Tuple[float, float]:
        mutation_prob = 1 - (std / max_std) if max_std > 0 and std != max_std else 0.5
        crossover_prob = 1 - mutation_prob
        return mutation_prob, crossover_prob

    def offspring_size(self, offspring_rate: float = None) -> int:
        if self.iterator.has_prev():
            num_of_new_individuals = self.iterator.prev()
            self.iterator.next()
        else:
            num_of_new_individuals = 1
        return num_of_new_individuals

    def get_main_metric(self, ind: Any) -> float:
        return ind.fitness.values[self.qual_position]

    def get_suppl_metric(self, ind: Any) -> float:
        return ind.fitness.values[self.compl_position]
