from typing import Any, Dict

from .interfaces.serializable import Serializable


class OperationSerializer(Serializable):
    """
    Serializer for "Operation" class

    Serialization: excludes "operations_repo" field cause it has no any important info about class
    Deserialization: uses basic method from superclass
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in super().to_json().items()
            if k != 'operations_repo'
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
