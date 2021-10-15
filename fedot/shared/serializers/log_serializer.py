from typing import Any, Dict

from ..interfaces.serializable import Serializable


class LogSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        basic_serialization = super().to_json()
        del basic_serialization['logger']  # cause it will be automatically generated in __init__
        return basic_serialization

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
