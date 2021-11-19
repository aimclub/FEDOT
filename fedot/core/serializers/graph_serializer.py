from itertools import product
from typing import Any, Dict

from .interfaces.serializable import Serializable


class GraphSerializer(Serializable):

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
