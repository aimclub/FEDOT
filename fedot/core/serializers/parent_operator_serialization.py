from typing import Any, Dict

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.opt_history import ParentOperator

from .any_serialization import any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    adapter = PipelineAdapter()
    serialized_op['parent_objects'] = [
        adapter.adapt(parent_obj.link_to_empty_pipeline)
        for parent_obj in serialized_op['parent_objects']
    ]
    return serialized_op
