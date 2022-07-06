from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from fedot.core.composer.advisor import DefaultChangeAdvisor
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_verifier import GraphVerifier, VerifierRuleType
from fedot.core.log import default_log
from fedot.core.adapter import BaseOptimizationAdapter, DirectAdapter, init_adapter
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, Objective, ObjectiveFunction
from fedot.core.optimisers.opt_node_factory import DefaultOptNodeFactory, OptNodeFactory

OptimisationCallback = Callable[[PopulationT, GenerationKeeper], Any]


def do_nothing_callback(*args, **kwargs):
    pass


@dataclass
class GraphOptimizerParameters:
    """Base class for definition of optimizer parameters. Can be extended for custom optimizers.

    :param multi_objective: defines if the optimizer must be multi-criterial
    :param offspring_rate: offspring rate used on next population
    :param pop_size: initial population size
    :param max_pop_size: maximum population size; optional, if unspecified, then population size is unbound
    :param adaptive_depth: flag to enable adaptive configuration of graph depth
    :param adaptive_depth_max_stagnation: max number of stagnating populations before adaptive depth increment
    """

    multi_objective: bool = False
    offspring_rate: float = 0.5
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    adaptive_depth: bool = False
    adaptive_depth_max_stagnation: int = 3


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
                 node_factory: Optional[OptNodeFactory] = None):
        self.adapter = adapter or DirectAdapter()
        self.verifier = GraphVerifier(rules_for_constraint)
        self.advisor = advisor or DefaultChangeAdvisor()
        self.node_factory = node_factory or DefaultOptNodeFactory()
        init_adapter(self.adapter)


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
    :param graph_optimizer_parameters: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Optional[Sequence[Graph]] = None,
                 requirements: Optional[ComposerRequirements] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_parameters: Optional[GraphOptimizerParameters] = None):
        self.log = default_log(self)
        self.initial_graphs = initial_graphs
        self._objective = objective
        self.requirements = requirements
        self.graph_generation_params = graph_generation_params or GraphGenerationParams()
        self.graph_optimizer_params = graph_optimizer_parameters or GraphOptimizerParameters()
        self._optimisation_callback: OptimisationCallback = do_nothing_callback

    @property
    def objective(self) -> Objective:
        return self._objective

    @abstractmethod
    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:
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
