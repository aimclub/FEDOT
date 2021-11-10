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
        vars(obj).update(**json_obj)
        return obj
