from typing import Any, Dict

from .interfaces.serializable import Serializable


class BasicSerializer(Serializable):
    '''
    Simple serializer which just uses abstract superclass interface implementation
    '''

    def to_json(self) -> Dict[str, Any]:
        return super().to_json()

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
