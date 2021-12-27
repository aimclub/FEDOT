from typing import Any, Dict

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import ParentOperator

from . import any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    serialized_op['parent_objects'] = [
        parent_obj.graph._serialization_id
        for parent_obj in serialized_op['parent_objects']
    ]
    return serialized_op
