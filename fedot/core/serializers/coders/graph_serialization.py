from typing import Any, Dict, Type, Sequence

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.linked_graph_node import LinkedGraphNode


def graph_from_json(cls: Type[Graph], json_obj: Dict[str, Any]) -> Graph:
    obj = cls()

    nodes_key = 'nodes' if 'nodes' in json_obj else '_nodes'
    if not issubclass(cls, GraphDelegate):
        nodes = json_obj[nodes_key]
        _reassign_edges_by_node_ids(nodes)
        obj.nodes = nodes

    # GraphDelegate case is handled by this
    vars(obj).update(**{k: v for k, v in json_obj.items() if k != nodes_key})

    return obj


def _reassign_edges_by_node_ids(nodes: Sequence[LinkedGraphNode]):
    """
    Assigns each <inner_node> from "nodes_from" to equal <outer_node> from "nodes"
        (cause each node from "nodes_from" in fact should point to the same node from "nodes")
    """
    lookup_dict = {node.uid: node for node in nodes}
    for node in nodes:
        if node.nodes_from:
            for parent_node_idx, parent_node_uid in enumerate(node.nodes_from):
                node.nodes_from[parent_node_idx] = lookup_dict.get(parent_node_uid, None)
