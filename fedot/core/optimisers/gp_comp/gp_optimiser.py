import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, List, Optional, Tuple, Union)

import numpy as np
from tqdm import tqdm
from deap.tools import ParetoFront

from fedot.core.composer.constraint import constraint_function
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.archive import SimpleArchive
from fedot.core.optimisers.gp_comp.gp_operators import (
    clean_operators_history,
    duplicates_filtration,
    evaluate_individuals,
    num_of_parents_in_crossover,
    random_graph
)
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, selection
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphOptimiser, GraphOptimiserParameters, correct_if_has_nans
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.optimisers.utils.population_utils import is_equal_archive, is_equal_fitness
from fedot.core.repository.quality_metrics_repository import MetricsEnum

MAX_NUM_OF_GENERATED_INDS = 10000
MIN_POPULATION_SIZE_WITH_ELITISM = 2


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
        :param multi_objective: flag used for of algorithm type definition (muti-objective if true or  single-objective
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
                 archive_type=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.selection_types = selection_types
        self.crossover_types = crossover_types
        self.mutation_types = mutation_types
        self.regularization_type = regularization_type
        self.genetic_scheme_type = genetic_scheme_type
        self.archive_type = archive_type


class EvoGraphOptimiser(GraphOptimiser):
    """
    Multi-objective evolutionary graph optimiser named GPComp
    """

    def __init__(self, initial_graph: Union[Any, List[Any]], requirements,
                 graph_generation_params: 'GraphGenerationParams',
                 metrics: List[MetricsEnum],
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 log: Log = None):

        super().__init__(initial_graph, requirements, graph_generation_params, metrics, parameters, log)

        self.graph_generation_params = graph_generation_params
        self.requirements = requirements

        self.parameters = GPGraphOptimiserParameters() if parameters is None else parameters
        self.parameters.set_default_params()
        if isinstance(self.parameters.archive_type, ParetoFront):
            self.archive = self.parameters.archive_type
        else:
            self.archive = SimpleArchive()

        self.max_depth = self.requirements.start_depth \
            if self.parameters.with_auto_depth_configuration and self.requirements.start_depth \
            else self.requirements.max_depth
        self.generation_num = 0
        self.num_of_gens_without_improvements = 0

        generation_depth = self.max_depth if self.requirements.start_depth is None else self.requirements.start_depth

        self.graph_generation_function = partial(random_graph, params=self.graph_generation_params,
                                                 requirements=self.requirements, max_depth=generation_depth)

        if not self.requirements.pop_size:
            self.requirements.pop_size = 10

        self.population = None
        self.initial_graph = initial_graph

    def _create_randomized_pop_from_inital_graph(self, initial_graphs: List[OptGraph]) -> List[Individual]:
        """
        Fill first population with mutated variants of the initial_graphs
        :param initial_graphs: Initial assumption for first population
        :return: list of individuals
        """
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        randomized_pop = []
        n_iter = self.requirements.pop_size * 10
        while n_iter > 0:
            initial_graph = np.random.choice(initial_graphs)
            n_iter -= 1
            new_ind = mutation(types=self.parameters.mutation_types,
                               params=self.graph_generation_params,
                               ind=Individual(deepcopy(initial_graph)),
                               requirements=initial_req,
                               max_depth=self.max_depth, log=self.log,
                               add_to_history=False)
            if new_ind not in randomized_pop:
                # to suppress duplicated
                randomized_pop.append(new_ind)

            if len(randomized_pop) == self.requirements.pop_size - len(initial_graphs):
                break

        # add initial graph to population
        for initial_graph in initial_graphs:
            randomized_pop.append(Individual(deepcopy(initial_graph)))

        return randomized_pop

    def _init_population(self):
        if self.initial_graph:
            adapted_graphs = [self.graph_generation_params.adapter.adapt(g) for g in self.initial_graph]
            self.population = self._create_randomized_pop_from_inital_graph(adapted_graphs)
        if self.population is None:
            self.population = self._make_population(self.requirements.pop_size)
        return self.population

    def optimise(self, objective_function, offspring_rate: float = 0.5,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:
        if on_next_iteration_callback is None:
            on_next_iteration_callback = self.default_on_next_iteration_callback

        self._init_population()

        num_of_new_individuals = self.offspring_size(offspring_rate)

        with OptimisationTimer(log=self.log, timeout=self.requirements.timeout) as t:
            pbar = tqdm(total=self.requirements.num_of_generations,
                        desc="Generations", unit='gen', initial=1) if show_progress else None

            self.population = self._evaluate_individuals(self.population, objective_function, timer=t)

            if self.archive is not None:
                self.archive.update(self.population)

            on_next_iteration_callback(self.population, self.archive)

            self.log_info_about_best()

            while (t.is_time_limit_reached(self.generation_num) is False
                   and self.generation_num != self.requirements.num_of_generations - 1):

                if self._is_stopping_criteria_triggered():
                    break

                self.log.info(f'Generation num: {self.generation_num}')

                self.num_of_gens_without_improvements = self.update_stagnation_counter()
                self.log.info(f'max_depth: {self.max_depth}, no improvements: {self.num_of_gens_without_improvements}')

                if self.parameters.with_auto_depth_configuration and self.generation_num != 0:
                    self.max_depth_recount()

                individuals_to_select = \
                    regularized_population(reg_type=self.parameters.regularization_type,
                                           population=self.population,
                                           objective_function=objective_function,
                                           graph_generation_params=self.graph_generation_params,
                                           timer=t)

                if self.parameters.multi_objective:
                    filtered_archive_items = duplicates_filtration(archive=self.archive,
                                                                   population=individuals_to_select)
                    individuals_to_select = deepcopy(individuals_to_select) + filtered_archive_items

                num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_parents,
                                                 params=self.graph_generation_params)

                new_population = []

                for parent_num in range(0, len(selected_individuals), 2):
                    new_population += self.reproduce(selected_individuals[parent_num],
                                                     selected_individuals[parent_num + 1])

                new_population = self._evaluate_individuals(new_population, objective_function, timer=t)

                self.prev_best = deepcopy(self.best_individual)

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop,
                                              graph_params=self.graph_generation_params)

                if not self.parameters.multi_objective and self.with_elitism:
                    self.population.append(self.prev_best)

                if self.archive is not None:
                    self.archive.update(self.population)

                on_next_iteration_callback(self.population, self.archive)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log_info_about_best()

                self.generation_num += 1

                if isinstance(self.archive, SimpleArchive):
                    self.archive.clear()

                clean_operators_history(self.population)

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            best = self.result_individual()
            self.log.info('Result:')
            self.log_info_about_best()

        output = [ind.graph for ind in best] if isinstance(best, list) else best.graph

        return output

    @property
    def best_individual(self) -> Any:
        if self.parameters.multi_objective:
            return self.archive
        else:
            return self.get_best_individual(self.population)

    @property
    def with_elitism(self) -> bool:
        if self.parameters.multi_objective:
            return False
        else:
            return self.requirements.pop_size > MIN_POPULATION_SIZE_WITH_ELITISM

    @property
    def num_of_inds_in_next_pop(self):
        return self.requirements.pop_size - 1 if self.with_elitism and not self.parameters.multi_objective \
            else self.requirements.pop_size

    def update_stagnation_counter(self) -> int:
        value = 0
        if self.generation_num != 0:
            if self.parameters.multi_objective:
                equal_best = is_equal_archive(self.prev_best, self.archive)
            else:
                equal_best = is_equal_fitness(self.prev_best.fitness, self.best_individual.fitness)
            if equal_best:
                value = self.num_of_gens_without_improvements + 1

        return value

    def log_info_about_best(self):
        if self.parameters.multi_objective:
            self.log.info(f'Pareto Frontier: '
                          f'{[item.fitness.values for item in self.archive.items if item.fitness is not None]}')
        else:
            self.log.info(f'Best metric is {self.best_individual.fitness}')

    def max_depth_recount(self):
        if self.num_of_gens_without_improvements == self.parameters.depth_increase_step and \
                self.max_depth + 1 <= self.requirements.max_depth:
            self.max_depth += 1

    def get_best_individual(self, individuals: List[Any], equivalents_from_current_pop=True) -> Any:
        inds_to_analyze = [ind for ind in individuals if ind.fitness is not None]
        best_ind = min(inds_to_analyze, key=lambda ind: ind.fitness)
        if equivalents_from_current_pop:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind)
        else:
            equivalents = self.simpler_equivalents_of_best_ind(best_ind, inds_to_analyze)

        if equivalents:
            best_candidate_id = min(equivalents, key=equivalents.get)
            best_ind = inds_to_analyze[best_candidate_id]
        return best_ind

    def simpler_equivalents_of_best_ind(self, best_ind: Any, inds: List[Any] = None) -> dict:
        individuals = self.population if inds is None else inds

        sort_inds = np.argsort([ind.fitness for ind in individuals])[1:]
        simpler_equivalents = {}
        for i in sort_inds:
            is_fitness_equals_to_best = is_equal_fitness(best_ind.fitness, individuals[i].fitness)
            has_less_num_of_operations_than_best = individuals[i].graph.length < best_ind.graph.length
            if is_fitness_equals_to_best and has_less_num_of_operations_than_best:
                simpler_equivalents[i] = len(individuals[i].graph.nodes)
        return simpler_equivalents

    def reproduce(self, selected_individual_first, selected_individual_second=None) -> Tuple[Any]:
        if selected_individual_second:
            new_inds = crossover(self.parameters.crossover_types,
                                 selected_individual_first,
                                 selected_individual_second,
                                 crossover_prob=self.requirements.crossover_prob,
                                 max_depth=self.max_depth, log=self.log,
                                 params=self.graph_generation_params)
        else:
            new_inds = [selected_individual_first]

        new_inds = tuple([mutation(types=self.parameters.mutation_types,
                                   params=self.graph_generation_params,
                                   ind=new_ind, requirements=self.requirements,
                                   max_depth=self.max_depth, log=self.log) for new_ind in new_inds])
        for ind in new_inds:
            ind.fitness = None
        return new_inds

    def _make_population(self, pop_size: int) -> List[Any]:
        pop = []
        iter_number = 0
        while len(pop) < pop_size:
            iter_number += 1
            graph = self.graph_generation_function()
            if constraint_function(graph, self.graph_generation_params):
                pop.append(Individual(graph))

            if iter_number > MAX_NUM_OF_GENERATED_INDS:
                self.log.debug(
                    f'More than {MAX_NUM_OF_GENERATED_INDS} generated in population making function. '
                    f'Process is stopped')
                break

        return pop

    def offspring_size(self, offspring_rate: float = None):
        default_offspring_rate = 0.5 if not offspring_rate else offspring_rate
        if self.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
            num_of_new_individuals = math.ceil(self.requirements.pop_size * default_offspring_rate)
        else:
            num_of_new_individuals = self.requirements.pop_size
        return num_of_new_individuals

    def default_on_next_iteration_callback(self, individuals, archive):
        try:
            self.history.add_to_history(individuals)
            self.history.save_current_results()
            archive = deepcopy(archive)
            if archive is not None:
                self.history.add_to_archive_history(archive.items)
        except Exception as ex:
            self.log.warn(f'Callback was not successful because of {ex}')

    def result_individual(self) -> Union[Any, List[Any]]:
        if not self.parameters.multi_objective:
            best = self.best_individual
        else:
            best = self.archive.items
        return best

    def _evaluate_individuals(self, individuals_set, objective_function, timer=None):
        evaluated_individuals = evaluate_individuals(individuals_set=individuals_set,
                                                     objective_function=objective_function,
                                                     graph_generation_params=self.graph_generation_params,
                                                     timer=timer, is_multi_objective=self.parameters.multi_objective)
        individuals_set = correct_if_has_nans(evaluated_individuals, self.log)
        return individuals_set

    def _is_stopping_criteria_triggered(self):
        is_stopping_needed = self.stopping_after_n_generation is not None
        if is_stopping_needed and self.num_of_gens_without_improvements == self.stopping_after_n_generation:
            self.log.info(f'GP_Optimiser: Early stopping criteria was triggered and composing finished')
            return True
        else:
            return False
