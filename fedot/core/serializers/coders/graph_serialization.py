from itertools import product
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
    for node in nodes:
        if node.nodes_from:
            for (idx, inner_node_uid), outer_node in product(enumerate(node.nodes_from), nodes):
                if inner_node_uid == outer_node.uid:
                    node.nodes_from[idx] = outer_node
    obj.nodes = nodes
    vars(obj).update(**{k: v for k, v in json_obj.items() if k != 'nodes'})
    return obj
