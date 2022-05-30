from abc import abstractmethod
from dataclasses import dataclass
from typing import (Any, Callable, List, Optional, Union, Sequence)

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.dag.graph import Graph
from fedot.core.log import Log, default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.generation_keeper import GenerationKeeper
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import Objective, ObjectiveEvaluate
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence

OptimisationCallback = Callable[[PopulationT, GenerationKeeper], None]


def do_nothing_callback(*args, **kwargs):
    pass


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

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Graph, Sequence[Graph]] = (),
                 requirements: Optional[Any] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 parameters: Optional[GraphOptimiserParameters] = None,
                 log: Optional[Log] = None):

        self.log = log or default_log(self.__class__.__name__)

        self._objective = objective
        self.requirements = requirements
        self.graph_generation_params = graph_generation_params or GraphGenerationParams()
        self.parameters = parameters or GraphOptimiserParameters()

        self.initial_graph = ensure_wrapped_in_sequence(initial_graph)

        # optimisation: callback function that runs on each iteration
        self.optimisation_callback: OptimisationCallback = do_nothing_callback

    @property
    def objective(self) -> Objective:
        return self._objective

    @abstractmethod
    def optimise(self, objective_evaluator: ObjectiveEvaluate,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:
        """
        Method for running of optimization using specified algorithm.
        :param objective_evaluator: Defines specific Objective and graph evaluation policy.
        :param show_progress: print output the describes the progress during iterations
        :return: best graph (or list of graph for multi-objective case)
        """
        pass
