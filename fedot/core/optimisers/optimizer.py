from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import (Any, Callable, List, Optional, Union, Sequence)

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.gp_comp.gp_operators import (random_graph)
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate


class GraphOptimiserParameters:
    """
        It is base class for defining the parameters of optimiser

        :param with_auto_depth_configuration: flag to enable option of automated tree depth configuration during
        evolution. Default False.
        :param depth_increase_step: the step of depth increase in automated depth configuration
        :param multi_objective: flag used for of algorithm type definition (multi-objective if true or single-objective
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
    Base class of graph optimiser. It allows to find the optimal solution using specified metric (one or several).
    To implement the specific optimisation method,
    the abstract method 'optimize' should be re-defined in the ancestor class
    (e.g. EvoGraphOptimiser, RandomSearchGraphOptimiser, etc).

    :param initial_graph: graph which was initialized outside the optimiser
    :param objective: objective for optimisation
    :param requirements: implementation-independent requirements for graph optimiser
    :param graph_generation_params: parameters for new graph generation
    :param parameters: parameters for specific implementation of graph optimiser
    :param log: optional parameter for log object
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 objective: Objective,
                 requirements: Any,
                 graph_generation_params: 'GraphGenerationParams',
                 parameters: GraphOptimiserParameters = None,
                 log: Optional[Log] = None):

        self.log = log or default_log(__name__)

        self._objective = objective
        self.graph_generation_params = graph_generation_params
        self.requirements = requirements
        self.parameters = parameters

        self.max_depth = self.requirements.start_depth \
            if self.requirements.start_depth \
            else self.requirements.max_depth

        self.graph_generation_function = partial(random_graph, params=self.graph_generation_params,
                                                 requirements=self.requirements, max_depth=self.max_depth)

        if initial_graph and not isinstance(initial_graph, Sequence):
            initial_graph = [initial_graph]
        self.initial_graph = initial_graph
        self.history = OptHistory(objective, parameters.history_folder)
        self.history.clean_results()

    @property
    def objective(self) -> Objective:
        return self._objective

    @abstractmethod
    def optimise(self, objective_evaluator: ObjectiveEvaluate,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:
        """
        Method for running of optimization using specified algorithm.
        :param objective_evaluator: Defines specific Objective and graph evaluation policy.
        :param on_next_iteration_callback: callback function that runs in each iteration of optimization
        :param show_progress: print output the describes the progress during iterations
        :return: best graph (or list of graph for multi-objective case)
        """
        pass

    def default_on_next_iteration_callback(self, individuals: Sequence[Individual],
                                           best_individuals: Optional[Sequence[Individual]] = None):
        """
        Default variant of callback that preserves optimisation history
        :param individuals: list of individuals obtained in last iteration
        :param best_individuals: optional list of the best individuals from all iterations
        :return:
        """
        try:
            self.history.add_to_history(individuals)
            if self.history.save_folder:
                self.history.save_current_results()
            if best_individuals is not None:
                self.history.add_to_archive_history(best_individuals)
        except Exception as ex:
            self.log.warn(f'Callback was not successful because of {ex}')


@dataclass
class GraphGenerationParams:
    """
    This dataclass is for defining the parameters using in graph generation process

    :param adapter: the function for processing of external object that should be optimized
    :param rules_for_constraint: collection of constraints
    :param advisor: class of task-specific advices for graph changes
    """
    adapter: BaseOptimizationAdapter = DirectAdapter()
    rules_for_constraint: Sequence[Callable] = tuple()
    advisor: Optional[DefaultChangeAdvisor] = DefaultChangeAdvisor()
