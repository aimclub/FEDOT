from abc import abstractmethod
from copy import deepcopy
from typing import Any, Type

from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate


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

    def restore_as_template(self, opt_graph: OptGraph):
        return self.restore(opt_graph)


class DirectAdapter(BaseOptimizationAdapter):
    def __init__(self, base_graph_class=None, base_node_class=None, log=None):
        """
        Optimization adapter for Pipeline class
        """
        base_node_class = base_node_class if base_node_class else OptNode
        base_graph_class = base_graph_class if base_graph_class else OptGraph

        super().__init__(base_graph_class=base_graph_class, base_node_class=base_node_class, log=log)

    def adapt(self, adaptee: Any):
        opt_graph = adaptee
        opt_graph.__class__ = OptGraph
        for node in opt_graph.nodes:
            _transform_node(node, OptNode)
        return opt_graph

    def restore(self, opt_graph: OptGraph):
        obj = opt_graph
        obj.__class__ = self._base_graph_class
        for node in opt_graph.nodes:
            _transform_node(node, self._base_node_class)
        return obj

    def restore_as_template(self, opt_graph: OptGraph):
        return self.restore(opt_graph)


class PipelineAdapter(BaseOptimizationAdapter):
    def __init__(self, log=None):
        """
        Optimization adapter for Pipeline class
        """
        super().__init__(base_graph_class=Pipeline, base_node_class=Node, log=log)

    def _transform_to_opt_node(self, node, *args, **kwargs):
        # Prepare content for nodes
        if not isinstance(node, OptNode):
            node.content = {'name': node.operation,
                            'params': node.custom_params}
            _transform_node(node=node, primary_class=OptNode,
                            transform_func=self._transform_to_opt_node)

    def _transform_to_pipeline_node(self, node, *args, **kwargs):
        _transform_node(node, PrimaryNode, SecondaryNode,
                        transform_func=self._transform_to_pipeline_node)
        if not node.nodes_from:
            node.__init__(operation_type=node.operation)
        else:
            node.__init__(nodes_from=node.nodes_from,
                          operation_type=node.operation)

    def adapt(self, adaptee: Pipeline):
        """ Convert Pipeline class into OptGraph class """
        source_pipeline = deepcopy(adaptee)

        # Apply recursive transformation since root
        self._transform_to_opt_node(source_pipeline.root_node)
        graph = OptGraph(source_pipeline.nodes)
        graph.uid = source_pipeline.uid
        return graph

    def restore(self, opt_graph: OptGraph):
        """ Convert OptGraph class into Pipeline class """
        source_graph = deepcopy(opt_graph)

        # TODO improve transformation
        # Inverse transformation since root node
        self._transform_to_pipeline_node(source_graph.root_node)
        pipeline = Pipeline(source_graph.nodes)
        pipeline.uid = source_graph.uid
        return pipeline

    def restore_as_template(self, opt_graph: OptGraph):
        pipeline = self.restore(opt_graph)
        return PipelineTemplate(pipeline)


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    raise ValueError('Parent node not in graph nodes list')


def _transform_node(node, primary_class, secondary_class=None, transform_func=None):
    if not transform_func:
        transform_func = _transform_node
    if not secondary_class:
        secondary_class = primary_class
    if node.nodes_from:
        node.__class__ = secondary_class
        for new_parent_node in node.nodes_from:
            transform_func(node=new_parent_node,
                           primary_class=primary_class,
                           secondary_class=secondary_class)
    else:
        node.__class__ = primary_class
