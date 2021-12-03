from typing import Any, Dict

from fedot.core.operations.operation import Operation

from . import any_serialization


def operation_to_json(obj: Operation) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "operations_repo" field cause it has no any important info about class
    """
    return {
        k: v
        for k, v in any_serialization.any_to_json(obj).items()
        if k != 'operations_repo'
    }
