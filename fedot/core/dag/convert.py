from typing import Tuple, Dict, TYPE_CHECKING
from uuid import uuid4

import networkx as nx

if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph
    from fedot.core.dag.graph_node import GraphNode


def graph_structure_as_nx_graph(structural_graph: 'Graph') -> Tuple[nx.DiGraph, Dict[uuid4, 'GraphNode']]:
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
            for parent in node.nodes_from:
                nx_graph.add_edge(new_node_indices[parent],
                                  new_node_indices[node])

    add_edges(nx_graph, structural_graph, new_node_indices)
    return nx_graph, node_labels
