from typing import Dict
from networkx import graph_edit_distance, set_node_attributes

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.convert import graph_structure_as_nx_graph


def get_distance_between(graph_1: Graph, graph_2: Graph, compare_node_params: bool = True) -> int:
    """
    Gets edit distance from ``graph_1`` graph to the ``graph_2``

    :param graph_1: left object to compare
    :param graph_2: right object to compare
    :compare_node_params: whether compare node internal params

    :return: graph edit distance (aka Levenstein distance for graphs)

    This function will expand golem.core.dag.linked_graph get_distance_between.
    """

    def node_match(node_data_1: Dict[str, GraphNode], node_data_2: Dict[str, GraphNode]) -> bool:
        """Checks if the two given nodes are identical

        :param node_data_1: nx_graph format for the first node to compare
        :param node_data_2: nx_graph format for the second node to compare

        :return: is the first node equal to the second
        """
        node_1, node_2 = node_data_1.get('node'), node_data_2.get('node')

        operations_do_match = str(node_1) == str(node_2)

        if not compare_node_params:
            return operations_do_match

        else:
            params_do_match = node_1.content.get('params') == node_2.content.get('params')
            nodes_do_match = operations_do_match and params_do_match

        return nodes_do_match

    graphs = (graph_1, graph_2)
    nx_graphs = []
    for graph in graphs:
        nx_graph, nodes = graph_structure_as_nx_graph(graph)
        set_node_attributes(nx_graph, nodes, name='node')
        nx_graphs.append(nx_graph)

    distance = graph_edit_distance(*nx_graphs, node_match=node_match)
    return int(distance)
