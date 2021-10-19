from typing import Any, Dict

from ..interfaces.serializable import Serializable


class GraphNodeSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        obj = super().to_json()
        del obj['_operator']  # to prevent circular references
        return obj

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
