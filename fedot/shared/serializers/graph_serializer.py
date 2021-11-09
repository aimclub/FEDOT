from typing import Any, Dict

from ..interfaces.serializable import Serializable


class GraphSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        useless_fields = set([
            'operator',  # to prevent circular reference
            'root_node', 'length', 'depth',
        ])
        return {
            k: v
            for k, v in super().to_json().items()
            if k not in useless_fields
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        obj = cls()
        vars(obj).update(**json_obj)
        return obj
