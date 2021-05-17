from abc import abstractmethod
from typing import Any, Type

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.chains.node import PrimaryNode, SecondaryNode, Node
from fedot.core.optimisers.graph import OptGraph, OptNode


class BaseOptimizationAdapter:
    def __init__(self, base_graph_class: Type, base_node_class: Type, log=None):
        """
        Base class for for the optimization adapter
        """
        self._log = log
        self._base_graph_class = base_graph_class
        self._base_node_class = base_node_class

    @abstractmethod
    def adapt(self, adaptee: Any):
        raise NotImplementedError()

    @abstractmethod
    def restore(self, opt_graph: OptGraph):
        raise NotImplementedError()

    @abstractmethod
    def restore_as_template(self, opt_graph: OptGraph):
        raise NotImplementedError()


class DirectAdapter(BaseOptimizationAdapter):
    def adapt(self, adaptee: Any):
        opt_graph = adaptee
        opt_graph.__class__ = OptGraph
        for node in opt_graph.nodes:
            node.__class = OptNode
        return opt_graph

    def restore(self, opt_graph: OptGraph):
        obj = opt_graph
        obj.__class__ = self._base_graph_class
        for node_id in range(len(opt_graph.nodes)):
            obj.nodes[node_id].__class__ = self._base_node_class
        return obj

    def restore_as_template(self, opt_graph: OptGraph):
        return self.restore(opt_graph)


class ChainAdapter(BaseOptimizationAdapter):
    def __init__(self, log=None):
        """
        Optimization adapter for Chain class
        """
        super().__init__(base_graph_class=Chain, base_node_class=Node, log=log)

    def adapt(self, adaptee: Chain):
        opt_nodes = []
        for node in adaptee.nodes:
            node.__class__ = OptNode
            opt_nodes.append(node)
        graph = OptGraph(opt_nodes)
        graph.uid = adaptee.uid
        return graph

    def restore(self, opt_graph: OptGraph):
        # TODO improve transformation
        chain_nodes = []
        for node in opt_graph.nodes:
            if node.nodes_from is None:
                node.__class__ = PrimaryNode
                node.__init__(operation_type=node.operation)
            else:
                node.__class__ = SecondaryNode
                node.__init__(nodes_from=node.nodes_from,
                              operation_type=node.operation)

            chain_nodes.append(node)
        chain = Chain(chain_nodes)
        chain.uid = opt_graph.uid
        return chain

    def restore_as_template(self, opt_graph: OptGraph):
        chain = self.restore(opt_graph)
        return ChainTemplate(chain)
