from typing import Any, Dict

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import ParentOperator

from . import any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    # adapter = PipelineAdapter()
    serialized_op['parent_objects'] = [
        parent_obj.unique_pipeline_id
        for parent_obj in serialized_op['parent_objects']
    ]
    return serialized_op
