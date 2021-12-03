from typing import Any, Dict

from fedot.core.optimisers.opt_history import ParentOperator

from . import any_serialization


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_serialization.any_to_json(obj)
    serialized_op['parent_objects'] = [
        parent_obj.link_to_empty_pipeline
        for parent_obj in serialized_op['parent_objects']
    ]
    return serialized_op
