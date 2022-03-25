from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, List, Optional, Union)

import numpy as np

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.gp_comp.gp_operators import (
    evaluate_individuals,
    random_graph
)
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.quality_metrics_repository import MetricsEnum


class GraphOptimiserParameters:
    """
        It is base class for defining the parameters of optimiser

        :param with_auto_depth_configuration: flag to enable option of automated tree depth configuration during
        evolution. Default False.
        :param depth_increase_step: the step of depth increase in automated depth configuration
        :param multi_objective: flag used for of algorithm type definition (muti-objective if true or  single-objective
        if false). Value is defined in ComposerBuilder. Default False.
    """

    def __init__(self,
                 with_auto_depth_configuration: bool = False, depth_increase_step: int = 3,
                 multi_objective: bool = False, history_folder: str = None,
                 stopping_after_n_generation: int = 10):
        self.with_auto_depth_configuration = with_auto_depth_configuration
        self.depth_increase_step = depth_increase_step
        self.multi_objective = multi_objective
        self.history_folder = history_folder
        self.stopping_after_n_generation = stopping_after_n_generation


class GraphOptimiser:
    """
    Base class of graph optimiser. It allow to find the optimal solution using specified metric (one or several).
    To implement the specific optimisation method,
    the abstract method 'optimize' should be re-defined in the ancestor class
    (e.g. EvoGraphOptimiser, RandomSearchGraphOptimiser, etc).

    :param initial_graph: graph which was initialized outside the optimiser
    :param requirements: implementation-independent requirements for graph optimizer
    :param graph_generation_params: parameters for new graph generation
    :param metrics: metrics for optimisation
    :param parameters: parameters for specific implementation of graph optimiser
    :param log: optional parameter for log object
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 requirements: Any,
                 graph_generation_params: 'GraphGenerationParams',
                 metrics: List[MetricsEnum],
                 parameters: GraphOptimiserParameters = None,
                 log: Log = None):

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self.graph_generation_params = graph_generation_params
        self.requirements = requirements

        self.max_depth = self.requirements.start_depth \
            if self.requirements.start_depth \
            else self.requirements.max_depth

        self.graph_generation_function = partial(random_graph, params=self.graph_generation_params,
                                                 requirements=self.requirements, max_depth=self.max_depth)

        self.stopping_after_n_generation = parameters.stopping_after_n_generation

        self.initial_graph = initial_graph
        self.history = OptHistory(metrics, parameters.history_folder)
        self.history.clean_results()

    @abstractmethod
    def optimise(self, objective_function,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:
        """
        Method for running of optimization using specified algorithm.
        :param objective_function: function for calculation of the objective function for optimisation
        :param on_next_iteration_callback: callback function that runs in each iteration of optimization
        :param show_progress: print output the describes the progress during iterations
        :return: best graph (or list of graph for multi-objective case)
        """
        pass

    def is_equal_fitness(self, first_fitness, second_fitness, atol=1e-10, rtol=1e-10) -> bool:
        """ Function for the comparison of fitness values between pairs of individuals
        :param first_fitness: fitness for individual A
        :param second_fitness: fitness for individual B
        :param atol: absolute tolerance parameter (see Notes).
        :param rtol: relative tolerance parameter (see Notes).
        :return: equality flag
        """
        return np.isclose(first_fitness, second_fitness, atol=atol, rtol=rtol)

    def default_on_next_iteration_callback(self, individuals: List[Individual],
                                           archive: Optional[List[Individual]] = None):
        """
        Default variant of callback that preserves optimisation history
        :param individuals: list of individuals obtained in iteration
        :param archive: optional list of best individuals for previous iterations
        :return:
        """
        try:
            self.history.add_to_history(individuals)
            if self.history.save_folder:
                self.history.save_current_results()
            archive = deepcopy(archive)
            if archive is not None:
                self.history.add_to_archive_history(archive.items)
        except Exception as ex:
            self.log.warn(f'Callback was not successful because of {ex}')

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


@dataclass
class GraphGenerationParams:
    """
    This dataclass is for defining the parameters using in graph generation process

    :param adapter: the function for processing of external object that should be optimized
    :param rules_for_constraint: set of constraints
    :param advisor: class of task-specific advices for graph changes
    """
    adapter: BaseOptimizationAdapter = DirectAdapter()
    rules_for_constraint: Optional[List[Callable]] = None
    advisor: Optional[DefaultChangeAdvisor] = DefaultChangeAdvisor()


def correct_if_has_nans(individuals, log):
    len_before = len(individuals)
    individuals = [ind for ind in individuals if ind.fitness is not None]
    len_after = len(individuals)

    if len_after == 0 and len_before != 0:
        raise ValueError('All evaluations of fitness were unsuccessful.')

    if len_after != len_before:
        log.info(f'None values were removed from candidates')

    return individuals
