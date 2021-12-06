from enum import Enum
from typing import Any, Dict, Type

from .. import Serializer


def enum_to_json(obj: Enum) -> Dict[str, Any]:
    return {
        'value': obj.value,
        **Serializer.dump_path_to_obj(obj)
    }


def enum_from_json(cls: Type[Enum], json_obj: Dict[str, Any]) -> Enum:
    return cls(json_obj['value'])
