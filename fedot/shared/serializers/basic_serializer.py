from typing import Any, Dict

from fedot.shared.interfaces.serializable import Serializable


class BasicSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        return super().to_json()

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
