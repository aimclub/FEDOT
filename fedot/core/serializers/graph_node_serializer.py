from typing import Any, Dict

from .interfaces.serializable import Serializable


class GraphNodeSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in super().to_json().items()
            if k != '_operator'  # to prevent circular references
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
