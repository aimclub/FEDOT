from typing import Any, Dict

from fedot.core.operations.operation import Operation

from . import any_to_json


def operation_to_json(obj: Operation) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "operations_repo" field cause it has no any important info about class
    """
    return {
        k: v
        for k, v in any_to_json(obj).items()
        if k not in ['operations_repo', '_eval_strategy', 'fitted_operation']
    }
