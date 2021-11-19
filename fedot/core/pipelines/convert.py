from typing import TYPE_CHECKING
from uuid import uuid4

import networkx as nx

if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph
    from fedot.core.pipelines.template import PipelineTemplate


def graph_structure_as_nx_graph(structural_graph: 'Graph'):
    """ Convert graph into networkx graph object """
    nx_graph = nx.DiGraph()
    node_labels = {}
    new_node_indices = {}
    for node in structural_graph.nodes:
        unique_id = uuid4()
        node_labels[unique_id] = node
        new_node_indices[node] = unique_id
        nx_graph.add_node(unique_id)

    def add_edges(nx_graph, structural_graph, new_node_indices):
        for node in structural_graph.nodes:
            if node.nodes_from is not None:
                for parent in node.nodes_from:
                    nx_graph.add_edge(new_node_indices[parent],
                                      new_node_indices[node])

    add_edges(nx_graph, structural_graph, new_node_indices)
    return nx_graph, node_labels


def pipeline_template_as_nx_graph(pipeline: 'PipelineTemplate'):
    """ Convert pipeline template into networkx graph object """
    graph = nx.DiGraph()
    node_labels = {}
    for operation in pipeline.operation_templates:
        unique_id, label = operation.operation_id, operation.operation_type
        node_labels[unique_id] = label
        graph.add_node(unique_id)

    def add_edges(graph, pipeline):
        for operation in pipeline.operation_templates:
            if operation.nodes_from is not None:
                for child in operation.nodes_from:
                    graph.add_edge(child, operation.operation_id)

    add_edges(graph, pipeline)
    return graph, node_labels
