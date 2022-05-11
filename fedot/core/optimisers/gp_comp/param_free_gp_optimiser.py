from itertools import zip_longest
from typing import List, Optional, Union, Callable, Any

import numpy as np
from tqdm import tqdm

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import (
    clean_operators_history,
    num_of_parents_in_crossover
)
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.iterator import SequenceIterator, fibonacci_sequence
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.parameters.population_size import PopulationSize, AdaptivePopulationSize
from fedot.core.optimisers.gp_comp.operators.regularization import regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import selection
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams, OptimisationCallback, do_nothing_cb
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate

DEFAULT_MAX_POP_SIZE = 55


class EvoGraphParameterFreeOptimiser(EvoGraphOptimiser):
    """
    Implementation of the parameter-free adaptive evolutionary optimiser
    (population size and genetic operators rates is changing over time).
    For details, see https://ieeexplore.ieee.org/document/9504773
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 objective: Objective,
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 max_population_size: int = DEFAULT_MAX_POP_SIZE,
                 log: Log = None):
        super().__init__(initial_graph, objective, requirements, graph_generation_params, parameters, log)

        self._min_population_size_with_elitism = 7
        if self.parameters.genetic_scheme_type != GeneticSchemeTypesEnum.parameter_free:
            self.log.warn(f'Invalid genetic scheme type was changed to parameter-free. Continue.')
            self.parameters.genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        pop_size_progression = SequenceIterator(sequence_func=fibonacci_sequence,
                                                start_value=requirements.pop_size,
                                                min_sequence_value=1, max_sequence_value=max_population_size)
        self._pop_size: PopulationSize = AdaptivePopulationSize(self.generations, pop_size_progression)

    def optimise(self, objective_evaluator: ObjectiveEvaluate,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:

        evaluator = self._get_evaluator(objective_evaluator)

        with self.timer, tqdm(total=self.requirements.num_of_generations,
                              desc='Generations', unit='gen', initial=1,
                              disable=not show_progress or self.log.verbosity_level == -1):

            pop_size = self._pop_size.initial
            self._next_population(evaluator(self._init_population(pop_size)))

            while not self.stop_optimisation():
                pop_size = self._pop_size.next(self.population)
                self.log.info(f'Next pop size: {pop_size}')

                # TODO: subst to mutation params
                if self.parameters.with_auto_depth_configuration and self.generations.generation_num > 0:
                    self.max_depth_recount()

                self.max_std = self.update_max_std()

                individuals_to_select = \
                    regularized_population(self.parameters.regularization_type,
                                           self.population,
                                           evaluator,
                                           self.graph_generation_params)

                # TODO: collapse this selection & reprodue for 1 and for many
                if len(self.population) == 1:
                    new_population = list(self.reproduce(self.population[0]))
                else:
                    num_of_parents = num_of_parents_in_crossover(pop_size)

                    selected_individuals = selection(types=self.parameters.selection_types,
                                                     population=individuals_to_select,
                                                     pop_size=num_of_parents,
                                                     params=self.graph_generation_params)

                    new_population = []
                    for ind_1, ind_2 in zip_longest(selected_individuals[::2], selected_individuals[1::2]):
                        new_population += self.reproduce(ind_1, ind_2)

                new_population = evaluator(new_population)

                with_elitism = self.with_elitism(pop_size)
                num_of_new_individuals = pop_size
                if with_elitism:
                    num_of_new_individuals -= len(self.generations.best_individuals)

                new_population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                             self.population,
                                             new_population, num_of_new_individuals,
                                             graph_params=self.graph_generation_params)

                # Add best individuals from the previous generation
                if with_elitism:
                    new_population.extend(self.generations.best_individuals)

                # Then update generation
                self._next_population(new_population)

                # TODO: move into dynamic mutation operator
                if not self.generations.is_any_improved:
                    self.operators_prob_update()

        best = self.generations.best_individuals
        return self.to_outputs(best)

    @property
    def current_std(self):
        return np.std([ind.fitness.value for ind in self.population])

    def update_max_std(self):
        if self.generations.generation_num <= 1:
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

    def operators_prob_update(self):
        std = float(self.current_std)
        max_std = float(self.max_std)

        mutation_prob = 1 - (std / max_std) if max_std > 0 and std != max_std else 0.5
        crossover_prob = 1 - mutation_prob

        self.requirements.mutation_prob = mutation_prob
        self.requirements.crossover_prob = crossover_prob
