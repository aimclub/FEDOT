from abc import abstractmethod
from copy import deepcopy
from typing import Any, Type

from fedot.core.dag.graph_node import GraphNode
from fedot.core.log import default_log
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate


class BaseOptimizationAdapter:
    def __init__(self, base_graph_class: Type, base_node_class: Type, log=None):
        """
        Base class for for the optimization adapter
        """
        self._log = log if log is not None else default_log('adapter_logger')
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
        self.base_node_class = base_node_class if base_node_class else OptNode
        self.base_graph_class = base_graph_class if base_graph_class else OptGraph

        super().__init__(base_graph_class=base_graph_class, base_node_class=base_node_class, log=log)

    def adapt(self, adaptee: Any):
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph

        for node in opt_graph.nodes:
            node.__class__ = OptNode
        return opt_graph

    def restore(self, opt_graph: OptGraph):
        obj = deepcopy(opt_graph)
        obj.__class__ = self.base_graph_class
        for node in obj.nodes:
            node.__class__ = self.base_node_class
        return obj

    def restore_as_template(self, opt_graph: OptGraph):
        return self.restore(opt_graph)


class PipelineAdapter(BaseOptimizationAdapter):
    def __init__(self, log=None):
        """
        Optimization adapter for Pipeline class
        """
        super().__init__(base_graph_class=Pipeline, base_node_class=Node, log=log)

    def _transform_to_opt_node(self, node, *args, **params):
        # Prepare content for nodes
        if type(node) == OptNode:
            self._log.warn('Unexpected: OptNode found in PipelineAdapter instead'
                           'PrimaryNode or SecondaryNode.')
        else:
            if type(node) == GraphNode:
                self._log.warn('Unexpected: GraphNode found in PipelineAdapter instead'
                               'PrimaryNode or SecondaryNode.')
            else:
                content = {'name': str(node.operation),
                           'params': node.custom_params}

                node.__class__ = OptNode
                node._fitted_operation = None
                node._node_data = None
                node.content = content

    def _transform_to_pipeline_node(self, node, *args, **params):
        if node.nodes_from:
            node.__class__ = params.get('secondary_class')
        else:
            node.__class__ = params.get('primary_class')
        if not node.nodes_from:
            node.__init__(operation_type=node.content['name'], content=node.content)
        else:
            node.__init__(nodes_from=node.nodes_from,
                          operation_type=node.content['name'], content=node.content
                          )

    def adapt(self, adaptee: Pipeline):
        """ Convert Pipeline class into OptGraph class """
        source_pipeline = deepcopy(adaptee)

        # Apply recursive transformation since root
        for node in source_pipeline.nodes:
            _transform_node(node=node, primary_class=OptNode,
                            transform_func=self._transform_to_opt_node)
        graph = OptGraph(source_pipeline.nodes)
        return graph

    def restore(self, opt_graph: OptGraph, computation_time=None):
        """ Convert OptGraph class into Pipeline class """
        source_graph = deepcopy(opt_graph)

        # Inverse transformation since root node
        for node in source_graph.nodes:
            _transform_node(node=node, primary_class=PrimaryNode, secondary_class=SecondaryNode,
                            transform_func=self._transform_to_pipeline_node)
        pipeline = Pipeline(source_graph.nodes)
        pipeline.computation_time = computation_time
        return pipeline

    def restore_as_template(self, opt_graph: OptGraph, computation_time=None):
        pipeline = self.restore(opt_graph, computation_time)
        tmp = PipelineTemplate(pipeline)
        return tmp


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    raise ValueError('Parent node not in graph nodes list')


def _transform_node(node, primary_class, secondary_class=None, transform_func=None):
    if transform_func:
        if not secondary_class:
            secondary_class = primary_class  # if there are no differences between primary and secondary class
        transform_func(node=node,
                       primary_class=primary_class,
                       secondary_class=secondary_class)
