from abc import abstractmethod
from copy import deepcopy
from typing import TypeVar, Generic, Type, Optional, Dict, Any

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph, OptNode

DomainStructureType = TypeVar('DomainStructureType')


class BaseOptimizationAdapter(Generic[DomainStructureType]):
    def __init__(self, base_graph_class: Type[DomainStructureType]):
        self._log = default_log(self)
        self.domain_graph_class = base_graph_class
        self.opt_graph_class = OptGraph

    def adapt(self, adaptee: DomainStructureType) -> OptGraph:
        if isinstance(adaptee, OptGraph):
            return adaptee
        return self._adapt(adaptee)

    def restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        if isinstance(opt_graph, self.domain_graph_class):
            return opt_graph
        return self._restore(opt_graph, metadata)

    def maybe_adapt(self, item):
        return self.adapt(item) if isinstance(item, self.domain_graph_class) else item

    def maybe_restore(self, item: OptGraph):
        return self.restore(item) if isinstance(item, self.opt_graph_class) else item

    def restore_ind(self, individual: Individual) -> DomainStructureType:
        return self.restore(individual.graph, individual.metadata)

    @abstractmethod
    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        raise NotImplementedError()

    @abstractmethod
    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        raise NotImplementedError()


class DirectAdapter(BaseOptimizationAdapter[DomainStructureType]):
    """Naive optimization adapter for arbitrary class that just overwrites __class__."""

    def __init__(self,
                 base_graph_class: Type[DomainStructureType] = OptGraph,
                 base_node_class: Type = OptNode):
        super().__init__(base_graph_class)
        self._base_node_class = base_node_class

    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph

        for node in opt_graph.nodes:
            node.__class__ = OptNode
        return opt_graph

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        obj = deepcopy(opt_graph)
        obj.__class__ = self.domain_graph_class
        for node in obj.nodes:
            node.__class__ = self._base_node_class
        return obj
