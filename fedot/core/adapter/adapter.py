from abc import abstractmethod
from copy import deepcopy
from typing import TypeVar, Generic, Type, Optional, Dict, Any

from fedot.core.log import default_log
from fedot.core.optimisers.graph import OptGraph, OptNode

AdapteeType = TypeVar('AdapteeType')
AdapteeNodeType = TypeVar('AdapteeNodeType')


class BaseOptimizationAdapter(Generic[AdapteeType, AdapteeNodeType]):
    def __init__(self,
                 base_graph_class: Type[AdapteeType],
                 base_node_class: Type[AdapteeNodeType]):
        self._log = default_log(self)
        self._base_graph_class = base_graph_class
        self._base_node_class = base_node_class

    def adapt(self, adaptee: AdapteeType) -> OptGraph:
        if isinstance(adaptee, OptGraph):
            return adaptee
        return self._adapt(adaptee)

    def restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> AdapteeType:
        if isinstance(opt_graph, self._base_graph_class):
            return opt_graph
        return self._restore(opt_graph, metadata)

    @abstractmethod
    def _adapt(self, adaptee: AdapteeType) -> OptGraph:
        raise NotImplementedError()

    @abstractmethod
    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> AdapteeType:
        raise NotImplementedError()

    def restore_as_template(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> AdapteeType:
        return self.restore(opt_graph, metadata)


class DirectAdapter(BaseOptimizationAdapter[AdapteeType, AdapteeNodeType]):
    """ Naive optimization adapter for arbitrary class that just overwrites __class__. """

    def __init__(self,
                 base_graph_class: Type[AdapteeType] = OptGraph,
                 base_node_class: Type[AdapteeNodeType] = OptNode):
        super().__init__(base_graph_class, base_node_class)

    def _adapt(self, adaptee: AdapteeType) -> OptGraph:
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph

        for node in opt_graph.nodes:
            node.__class__ = OptNode
        return opt_graph

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> AdapteeType:
        obj = deepcopy(opt_graph)
        obj.__class__ = self._base_graph_class
        for node in obj.nodes:
            node.__class__ = self._base_node_class
        return obj
