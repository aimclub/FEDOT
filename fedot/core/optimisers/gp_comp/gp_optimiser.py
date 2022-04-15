import math
from copy import deepcopy
from functools import partial
from itertools import zip_longest
from typing import Any, Callable, Optional, Tuple, Union, \
    List, Sequence, Iterable, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from fedot.core.composer.constraint import constraint_function
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import (
    clean_operators_history,
    duplicates_filtration,
    num_of_parents_in_crossover,
    random_graph
)
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.operators.evaluation import Evaluate
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.generation_keeper import best_individual, GenerationKeeper
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum, regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, selection
from fedot.core.optimisers.gp_comp.composite_condition import CompositeCondition
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.optimisers.utils.population_utils import is_equal_archive
from fedot.core.repository.quality_metrics_repository import MetricsEnum

if TYPE_CHECKING:
    from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements

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

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 requirements: 'PipelineComposerRequirements',
                 graph_generation_params: GraphGenerationParams,
                 metrics: List[MetricsEnum],
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 log: Optional[Log] = None):

        super().__init__(initial_graph, requirements, graph_generation_params, metrics, parameters, log)

        self.graph_generation_params = graph_generation_params
        self.requirements = requirements

        self.parameters = GPGraphOptimiserParameters() if parameters is None else parameters
        self.parameters.set_default_params()

        self.max_depth = self.requirements.start_depth \
            if self.parameters.with_auto_depth_configuration and self.requirements.start_depth \
            else self.requirements.max_depth
        generation_depth = self.requirements.start_depth or self.max_depth
        self.graph_generation_function = partial(random_graph, params=self.graph_generation_params,
                                                 requirements=self.requirements, max_depth=generation_depth)

        if not self.requirements.pop_size:
            self.requirements.pop_size = 10

        self.population = None
        self.initial_graph = initial_graph

        self.generations = GenerationKeeper(initial_generation=None,  # TODO: either remove or ensure it can be passed
                                            is_multi_objective=self.parameters.multi_objective,
                                            # TODO: ensure if archive_type is instance or cls itself
                                            archive=self.parameters.archive_type)

        self.timer = OptimisationTimer(timeout=self.requirements.timeout, log=self.log)
        objective_function = None  # TODO: pass it
        intermediate_metrics_function = None  # TODO: pass it through init
        n_jobs = self.requirements.n_jobs
        self.evaluator = Evaluate(graph_gen_params=graph_generation_params,
                                  objective_function=objective_function,
                                  is_multi_objective=self.parameters.multi_objective,
                                  timer=self.timer, log=self.log, n_jobs=n_jobs)

        # stopping_after_n_generation may be None, so use some obvious max number
        max_stagnation_length = parameters.stopping_after_n_generation or self.requirements.num_of_generations
        self.stop_optimisation = \
            CompositeCondition(self.log) \
            .add_condition(
                lambda: self.timer.is_time_limit_reached(self.generations.generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.generations.generation_num >= self.requirements.num_of_generations,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_length >= max_stagnation_length,
                'Optimisation finished: Early stopping criteria was satisfied'
            )

    def _create_randomized_pop(self, individuals: List[Individual]) -> List[Individual]:
        """
        Fill first population with mutated variants of the initial_graphs
        :param individuals: Initial assumption for first population
        :return: list of individuals
        """
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        randomized_pop = []
        n_iter = self.requirements.pop_size * 10
        while n_iter > 0:
            initial_individual = np.random.choice(individuals)
            n_iter -= 1
            new_ind = mutation(types=self.parameters.mutation_types,
                               params=self.graph_generation_params,
                               ind=initial_individual,
                               requirements=initial_req,
                               max_depth=self.max_depth, log=self.log)
            if new_ind not in randomized_pop:
                # to suppress duplicated
                randomized_pop.append(new_ind)

            if len(randomized_pop) == self.requirements.pop_size - len(individuals):
                break

        # add initial graph to population
        for initial in individuals:
            randomized_pop.append(initial)

        return randomized_pop

    def _init_population(self):
        if self.initial_graph:
            initial_individuals = [Individual(self.graph_generation_params.adapter.adapt(g)) for g in
                                   self.initial_graph]
            self.population = self._create_randomized_pop(initial_individuals)
        if self.population is None:
            self.population = self._make_population(self.requirements.pop_size)
        return self.population

    # TODO: fix invalid signature according to base method (`offspring_rate` is a new param)
    def optimise(self, objective_function, offspring_rate: float = 0.5,
                 on_next_iteration_callback: Optional[Callable] = None,
                 intermediate_metrics_function: Optional[Callable] = None,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:

        self.evaluator.objective_function = objective_function  # TODO: move into init!
        self.evaluator._intermediate_metrics_function = intermediate_metrics_function  # TODO: move into init!

        if on_next_iteration_callback is None:
            on_next_iteration_callback = self.default_on_next_iteration_callback

        num_of_new_individuals = self.offspring_size(offspring_rate)
        self.log.info(f'Number of new individuals: {num_of_new_individuals}')

        with self.timer as t:
            pbar = tqdm(total=self.requirements.num_of_generations,
                        desc='Generations', unit='gen', initial=1,
                        disable=self.log.verbosity_level == -1) if show_progress else None

            self._init_population()
            self.population = self.evaluator(self.population)
            self.generations.append(self.population)

            on_next_iteration_callback(self.population, self.generations.best_individuals)
            self.log_info_about_best()

            while not self.stop_optimisation():
                self.log.info(f'Generation num: {self.generations.generation_num}')
                self.log.info(f'max_depth: {self.max_depth}, no improvements: {self.generations.stagnation_length}')

                # TODO: subst to mutation params
                if self.parameters.with_auto_depth_configuration and self.generations.generation_num > 0:
                    self.max_depth_recount()

                individuals_to_select = \
                    regularized_population(reg_type=self.parameters.regularization_type,
                                           population=self.population,
                                           objective_function=objective_function,
                                           graph_generation_params=self.graph_generation_params,
                                           timer=t)

                if self.parameters.multi_objective:
                    # TODO: feels unneeded, ParetoFront does it anyway
                    filtered_archive_items = duplicates_filtration(self.generations.best_individuals, individuals_to_select)
                    individuals_to_select = deepcopy(individuals_to_select) + filtered_archive_items

                num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                selected_individuals = selection(types=self.parameters.selection_types,
                                                 population=individuals_to_select,
                                                 pop_size=num_of_parents,
                                                 params=self.graph_generation_params)

                new_population = []

                for ind_1, ind_2 in zip_longest(selected_individuals[::2], selected_individuals[1::2]):
                    new_population += self.reproduce(ind_1, ind_2)

                new_population = self.evaluator(new_population)

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop,
                                              graph_params=self.graph_generation_params)

                # Add best individuals from the previous generation
                if not self.parameters.multi_objective and self.with_elitism:
                    self.population.extend(self.generations.best_individuals)
                # Then update generation
                self.generations.append(self.population)

                on_next_iteration_callback(self.population, self.generations.best_individuals)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log_info_about_best()

                clean_operators_history(self.population)

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            best = self.generations.best_individuals
            self.log.info('Result:')
            self.log_info_about_best()

        return self.to_outputs(best)

    def to_outputs(self, individuals: Iterable[Individual]) -> Union[OptGraph, List[OptGraph]]:
        # TODO: switch to uniform interface, always return list
        #  because if the caller needs single output -- it can just take it
        graphs = [ind.graph for ind in individuals]
        # for single objective with single result return it directly
        if not self.parameters.multi_objective and len(graphs) == 1:
            return graphs[0]
        return graphs

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

    def log_info_about_best(self):
        best = self.generations.best_individuals
        if self.parameters.multi_objective:
            self.log.info(f'Pareto Frontier: '
                          f'{[item.fitness.values for item in best]}')
        else:
            self.log.info(f'Best metric is {best[0].fitness}')

    def max_depth_recount(self):
        if self.generations.stagnation_length >= self.parameters.depth_increase_step and \
                self.max_depth + 1 <= self.requirements.max_depth:
            self.max_depth += 1

    def reproduce(self,
                  selected_individual_first: Individual,
                  selected_individual_second: Optional[Individual] = None) -> Tuple[Any]:

        selected_individual_first.parent_operators = []

        if selected_individual_second:
            selected_individual_second.parent_operators = []
            new_inds = crossover(self.parameters.crossover_types,
                                 selected_individual_first,
                                 selected_individual_second,
                                 crossover_prob=self.requirements.crossover_prob,
                                 max_depth=self.max_depth, log=self.log,
                                 params=self.graph_generation_params)
        else:
            new_inds = [selected_individual_first]

        new_inds = [mutation(types=self.parameters.mutation_types,
                             params=self.graph_generation_params,
                             ind=new_ind, requirements=self.requirements,
                             max_depth=self.max_depth, log=self.log) for new_ind in new_inds]
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
