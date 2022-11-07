from typing import Any, Dict

from fedot.core.dag.linked_graph_node import LinkedGraphNode
from .. import any_to_json


def graph_node_to_json(obj: LinkedGraphNode) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "_operator" field to rid of circular references
    """
    encoded = {
        k: v
        for k, v in any_to_json(obj).items()
        if k not in ['_operator', '_fitted_operation', '_node_data', '_parameters']
    }
    encoded['content']['name'] = str(encoded['content']['name'])
    if encoded['_nodes_from']:
        encoded['_nodes_from'] = [
            node.uid
            for node in encoded['_nodes_from']
        ]
    return encoded
