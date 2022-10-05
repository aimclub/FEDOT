from typing import List, Optional, Callable, Sequence

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode


def nodes_same(left_nodes: Sequence[GraphNode], right_nodes: Sequence[GraphNode]) -> bool:
    left_set = set(map(lambda n: n.descriptive_id, left_nodes))
    right_set = set(map(lambda n: n.descriptive_id, right_nodes))
    return left_set == right_set and len(left_nodes) == len(right_nodes)


def graphs_same(left: Graph, right: Graph) -> bool:
    return left == right


def find_same_node(nodes: List[GraphNode], target: GraphNode) -> Optional[GraphNode]:
    return next(filter(lambda n: n.descriptive_id == target.descriptive_id, nodes), None)


def find_first(graph, predicate: Callable[[GraphNode], bool]) -> Optional[GraphNode]:
    return next(filter(predicate, graph.nodes), None)
