from copy import deepcopy
from typing import Any, Optional, Dict

from fedot.core.adapter import BaseOptimizationAdapter
from fedot.core.dag.graph_node import LinkedGraphNode
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


class PipelineAdapter(BaseOptimizationAdapter[Pipeline]):
    """ Optimization adapter for Pipeline class """

    def __init__(self):
        super().__init__(base_graph_class=Pipeline)

    def _transform_to_opt_node(self, node, *args, **params):
        # Prepare content for nodes
        if type(node) == OptNode:
            self._log.warning('Unexpected: OptNode found in PipelineAdapter instead'
                              'PrimaryNode or SecondaryNode.')
        else:
            if type(node) == LinkedGraphNode:
                self._log.warning('Unexpected: GraphNode found in PipelineAdapter instead'
                                  'PrimaryNode or SecondaryNode.')
            else:
                content = {'name': str(node.operation),
                           'params': node.parameters,
                           'metadata': node.metadata}

                node.__class__ = OptNode
                node._fitted_operation = None
                node._node_data = None
                del node.metadata
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

    def _adapt(self, adaptee: Pipeline) -> OptGraph:
        """ Convert Pipeline class into OptGraph class """
        source_pipeline = deepcopy(adaptee)

        # Apply recursive transformation since root
        for node in source_pipeline.nodes:
            _transform_node(node=node, primary_class=OptNode,
                            transform_func=self._transform_to_opt_node)
        graph = OptGraph(source_pipeline.nodes)
        return graph

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> Pipeline:
        """ Convert OptGraph class into Pipeline class """
        metadata = metadata or {}
        source_graph = deepcopy(opt_graph)

        # Inverse transformation since root node
        for node in source_graph.nodes:
            _transform_node(node=node, primary_class=PrimaryNode, secondary_class=SecondaryNode,
                            transform_func=self._transform_to_pipeline_node)
        pipeline = Pipeline(source_graph.nodes)
        pipeline.computation_time = metadata.get('computation_time_in_seconds')
        return pipeline


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
