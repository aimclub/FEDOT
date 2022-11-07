from typing import Any, Dict, Type

from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from .. import any_from_json, any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    serialized_op['parent_individuals'] = [
        parent_ind.uid
        for parent_ind in serialized_op['parent_individuals'] if parent_ind is not None
    ]
    return serialized_op


def parent_operator_from_json(cls: Type[ParentOperator], json_obj: Dict[str, Any]) -> ParentOperator:
    deserialized = any_from_json(cls, json_obj)
    object.__setattr__(deserialized, 'parent_individuals', deserialized.parent_individuals)
    return deserialized
