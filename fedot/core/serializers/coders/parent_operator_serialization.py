from typing import Any, Dict

from fedot.core.optimisers.opt_history import ParentOperator
from . import any_to_json


def parent_operator_to_json(obj: ParentOperator) -> Dict[str, Any]:
    serialized_op = any_to_json(obj)
    serialized_op['parent_individuals'] = [
        parent_ind.uid
        for parent_ind in serialized_op['parent_individuals'] if parent_ind is not None
    ]
    return serialized_op
