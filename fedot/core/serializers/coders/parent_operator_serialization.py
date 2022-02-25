from typing import Any, Dict

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.opt_history import ParentOperator

from . import any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    serialized_op['parent_objects'] = [
        parent_obj.graph._serialization_id
        for parent_obj in serialized_op['parent_objects']
        if isinstance(parent_obj, Individual)
    ]
    return serialized_op
