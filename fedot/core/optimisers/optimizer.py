from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import (Any, Callable, Optional, Sequence)

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_verifier import GraphVerifier, VerifierRuleType
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import Objective, ObjectiveFunction, GraphFunction
from fedot.core.optimisers.opt_node_factory import OptNodeFactory, DefaultOptNodeFactory

OptimisationCallback = Callable[[PopulationT, GenerationKeeper], Any]


def do_nothing_callback(*args, **kwargs):
    pass


class GraphOptimizerParameters:
    """
        It is base class for defining the parameters of optimizer

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
        self.stopping_after_n_generation = stopping_after_n_generation


@dataclass
class GraphGenerationParams:
    """
    This dataclass is for defining the parameters using in graph generation process

    :param adapter: the function for processing of external object that should be optimized
    :param rules_for_constraint: collection of constraints
    :param advisor: class of task-specific advices for graph changes
    :param node_factory: class of generating nodes while mutation
    """
    adapter: BaseOptimizationAdapter
    verifier: GraphVerifier
    advisor: DefaultChangeAdvisor
    node_factory: OptNodeFactory

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None,
                 rules_for_constraint: Sequence[VerifierRuleType] = (),
                 advisor: Optional[DefaultChangeAdvisor] = None,
                 node_factory: OptNodeFactory = None):
        self.adapter = adapter or DirectAdapter()
        self.verifier = GraphVerifier(rules_for_constraint, self.adapter)
        self.advisor = advisor or DefaultChangeAdvisor()
        self.node_factory = node_factory or DefaultOptNodeFactory()


class GraphOptimizer:
    """
    Base class of graph optimizer. It allows to find the optimal solution using specified metric (one or several).
    To implement the specific optimisation method,
    the abstract method 'optimize' should be re-defined in the ancestor class
    (e.g.  PopulationalOptimizer, RandomSearchGraphOptimiser, etc).

    :param objective: objective for optimisation
    :param initial_graphs: graphs which were initialized outside the optimizer
    :param requirements: implementation-independent requirements for graph optimizer
    :param graph_generation_params: parameters for new graph generation
    :param parameters: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Optional[Sequence[Graph]] = None,
                 requirements: Optional[Any] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 parameters: Optional[GraphOptimizerParameters] = None):
        self.log = default_log(self, logging_level=requirements.logging_level_opt if requirements else
                               logging.CRITICAL+1)
        self.initial_graphs = initial_graphs
        self._objective = objective
        self.requirements = requirements
        self.graph_generation_params = graph_generation_params or GraphGenerationParams()
        self.parameters = parameters or GraphOptimizerParameters()
        self._optimisation_callback: OptimisationCallback = do_nothing_callback

    @property
    def objective(self) -> Objective:
        return self._objective

    @abstractmethod
    def optimise(self, objective: ObjectiveFunction) -> PopulationT:
        """
        Method for running of optimization using specified algorithm.
        :param objective: objective function that specifies optimization target
        :return: sequence of the best graphs
        """
        pass

    def set_optimisation_callback(self, callback: Optional[OptimisationCallback]):
        """Set optimisation callback that must run on each iteration.
        Reset the callback if None is passed."""
        self._optimisation_callback = callback or do_nothing_callback

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        """Set or reset (with None) post-evaluation callback
        that's called on each graph after its evaluation."""
        pass
