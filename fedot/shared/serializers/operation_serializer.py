from typing import Any, Dict

from ..interfaces.serializable import Serializable


class OperationSerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        useless_fields = set([
            'operations_repo', 'metadata'
        ])
        return {
            k: v
            for k, v in super().to_json().items()
            if k not in useless_fields
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
