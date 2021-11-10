from typing import Any, Dict

from ..interfaces.serializable import Serializable


class GraphSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in super().to_json().items()
            if k != 'operator'  # to prevent circular reference
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        obj = cls()
        nodes = json_obj['nodes']
        for node in nodes:
            if node.nodes_from:
                for j, inner_node in enumerate(node.nodes_from):
                    for node_outer in nodes:
                        if inner_node.descriptive_id == node_outer.descriptive_id:
                            node.nodes_from[j] = node_outer
                            break
        obj.nodes = nodes
        vars(obj).update(**{k: v for k, v in json_obj.items() if k != 'nodes'})
        return obj
