from itertools import product
from typing import Any, Dict

from .interfaces.serializable import Serializable


class GraphSerializer(Serializable):
    '''
    Serializer for "Graph" class

    Serialization: excludes "operator" field to rid of circular references
        also saves idx of each node from 'nodes' field to simplify deserialization
    Deserialization: assigns each <inner_node> from "nodes_from" to equal <outer_node> from "nodes"
        (cause each node from "nodes_from" in fact should point to the same node from "nodes")
    '''

    def to_json(self) -> Dict[str, Any]:
        serialized_obj = {
            k: v
            for k, v in super().to_json().items()
            if k != 'operator'  # to prevent circular reference
        }
        for idx, node in enumerate(serialized_obj['nodes']):
            node._serialization_id = idx
        return serialized_obj

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        obj = cls()
        nodes = json_obj['nodes']
        for node in nodes:
            if node.nodes_from:
                for (idx, inner_node), outer_node in product(enumerate(node.nodes_from), nodes):
                    if inner_node._serialization_id == outer_node._serialization_id:
                        node.nodes_from[idx] = outer_node
        obj.nodes = nodes
        vars(obj).update(**{k: v for k, v in json_obj.items() if k != 'nodes'})
        return obj
