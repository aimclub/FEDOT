from typing import Any, Dict, Type
from uuid import UUID


def uuid_to_json(obj: UUID) -> Dict[str, Any]:
    return { 'hex': obj.hex }


def uuid_from_json(cls: Type[UUID], json_obj: Dict[str, Any]) -> UUID:
    return cls(json_obj['hex'])
