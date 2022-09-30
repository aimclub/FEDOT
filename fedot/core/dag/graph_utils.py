from collections.abc import Sequence
from typing import Union, List

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


def distance_to_root_level(graph: Graph, node: GraphNode) -> int:
    """Gets distance to the final output node

    Args:
        node: search starting point

    Return:
        int: distance to root level
    """

    def recursive_child_height(parent_node: GraphNode) -> int:
        """Recursively dives into ``parent_node`` children to get the bottom height

        :param node: search starting point
        """
        node_child = graph.node_children(parent_node)
        if node_child:
            height = recursive_child_height(node_child[0]) + 1
            return height
        return 0

    height = recursive_child_height(node)
    return height


def nodes_from_layer(graph: Graph, layer_number: int) -> Sequence[GraphNode]:
    """Gets all the nodes from the chosen layer up to the surface

    Args:
        layer_number: max height of diving

    Returns:
        all nodes from the surface to the ``layer_number`` layer
    """

    def get_nodes(node: Union[GraphNode, List[GraphNode]], current_height: int):
        """Gets all the parent nodes of ``node``

        :param node: node to get all subnodes from
        :param current_height: current diving step depth

        :return: all parent nodes of ``node``
        """
        nodes = []
        if current_height == layer_number:
            nodes.append(node)
        else:
            if node.nodes_from:
                for child in node.nodes_from:
                    nodes.extend(get_nodes(child, current_height + 1))
        return nodes

    nodes = get_nodes(graph.root_node, current_height=0)
    return nodes
