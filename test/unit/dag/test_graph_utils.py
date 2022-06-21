from typing import Tuple, List, Optional, Iterable

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode


def nodes_same(left_nodes: Iterable[GraphNode], right_nodes: Iterable[GraphNode]) -> bool:
    left_set = set(map(lambda n: n.descriptive_id, left_nodes))
    right_set = set(map(lambda n: n.descriptive_id, right_nodes))
    return left_set == right_set


def graphs_same(left: Graph, right: Graph) -> bool:
    return nodes_same(left.nodes, right.nodes)


def find_same_node(nodes: List[GraphNode], target: GraphNode) -> Optional[GraphNode]:
    return next(filter(lambda n: n.descriptive_id == target.descriptive_id, nodes), None)
