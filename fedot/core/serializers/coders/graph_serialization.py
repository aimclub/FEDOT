from typing import Any, Dict, Type

from fedot.core.dag.graph import Graph
from . import any_to_json


def graph_to_json(obj: Graph) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "operator" field to rid of circular references
    """
    serialized_obj = {
        k: v
        for k, v in any_to_json(obj).items()
        if k != 'operator'  # to prevent circular reference
    }
    return serialized_obj


def graph_from_json(cls: Type[Graph], json_obj: Dict[str, Any]) -> Graph:
    """
    Assigns each <inner_node> from "nodes_from" to equal <outer_node> from "nodes"
        (cause each node from "nodes_from" in fact should point to the same node from "nodes")
    """
    obj = cls()
    nodes = json_obj['nodes']

    lookup_dict = {node.uid: node for node in nodes}

    for node in nodes:
        if node.nodes_from:
            for parent_node_idx, parent_node_uid in enumerate(node.nodes_from):
                node.nodes_from[parent_node_idx] = lookup_dict.get(parent_node_uid, None)
    obj.nodes = nodes
    vars(obj).update(**{k: v for k, v in json_obj.items() if k != 'nodes'})
    return obj
