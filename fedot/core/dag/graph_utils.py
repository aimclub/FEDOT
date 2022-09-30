from collections.abc import Sequence

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.pipelines.convert import graph_structure_as_nx_graph


def get_nodes_degrees(graph: Graph) -> Sequence[int]:
    """Nodes degree as the number of edges the node has:
        ``degree = #input_edges + #out_edges``

    Returns:
        nodes degrees ordered according to the nx_graph representation of this graph
    """
    graph, _ = graph_structure_as_nx_graph(graph)
    index_degree_pairs = graph.degree
    node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
    return node_degrees
