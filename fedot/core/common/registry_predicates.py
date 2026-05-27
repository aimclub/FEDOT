from typing import Any
from fedot.core.data.common.types import ARRAY_RUNTIME_TYPES
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.planner import PreprocessingPlan
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
import inspect


__all__ = [
    'is_array_runtime',
    'is_tensor_data',
    'is_preprocessing_plan',
    'is_preprocessing_handler',
]

def is_array_runtime(data: Any) -> bool:
    return isinstance(data, ARRAY_RUNTIME_TYPES)


def is_tensor_data(data: Any) -> bool:
    return isinstance(data, TensorData)


def is_preprocessing_plan(data: Any) -> bool:
    return isinstance(data, PreprocessingPlan)


def is_preprocessing_handler(data: Any) -> bool:
    if inspect.isclass(data):
        return issubclass(data, AbstractPreprocessingHandler)

    return isinstance(data, AbstractPreprocessingHandler)
