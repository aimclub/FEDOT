from abc import abstractmethod
from typing import Any, Type

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode, Node
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


class PipelineAdapter(BaseOptimizationAdapter):
    def __init__(self, log=None):
        """
        Optimization adapter for Pipeline class
        """
        super().__init__(base_graph_class=Pipeline, base_node_class=Node, log=log)

    def adapt(self, adaptee: Pipeline):
        opt_nodes = []
        for node in adaptee.nodes:
            node.__class__ = OptNode
            opt_nodes.append(node)
        graph = OptGraph(opt_nodes)
        graph.uid = adaptee.uid
        return graph

    def restore(self, opt_graph: OptGraph):
        # TODO improve transformation
        pipeline_nodes = []
        for node in opt_graph.nodes:
            if node.nodes_from is None:
                node.__class__ = PrimaryNode
                node.__init__(operation_type=node.operation)
            else:
                node.__class__ = SecondaryNode
                node.__init__(nodes_from=node.nodes_from,
                              operation_type=node.operation)

            pipeline_nodes.append(node)
        pipeline = Pipeline(pipeline_nodes)
        pipeline.uid = opt_graph.uid
        return pipeline

    def restore_as_template(self, opt_graph: OptGraph):
        pipeline = self.restore(opt_graph)
        return PipelineTemplate(pipeline)
