from typing import Any, Dict

from .interfaces.serializable import Serializable


class GraphNodeSerializer(Serializable):
    """
    Serializer for "GraphNode" class

    Serialization: excludes "_operator" field to rid of circular references
    Deserialization: uses basic method from superclass
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in super().to_json().items()
            if k != '_operator'  # to prevent circular references
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
