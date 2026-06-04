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
    """
    Return whether ``data`` is a supported in-memory feature array.

    Args:
        data: Candidate runtime object.

    Returns:
        ``True`` for NumPy/CuPy arrays registered in ``ARRAY_RUNTIME_TYPES``.
    """
    return isinstance(data, ARRAY_RUNTIME_TYPES)


def is_tensor_data(data: Any) -> bool:
    """
    Return whether ``data`` is a ``TensorData`` instance.

    Args:
        data: Candidate runtime object.

    Returns:
        ``True`` when ``data`` is ``TensorData``.
    """
    return isinstance(data, TensorData)


def is_preprocessing_plan(data: Any) -> bool:
    """
    Return whether ``data`` is a ``PreprocessingPlan`` instance.

    Args:
        data: Candidate runtime object.

    Returns:
        ``True`` when ``data`` is ``PreprocessingPlan``.
    """
    return isinstance(data, PreprocessingPlan)


def is_preprocessing_handler(data: Any) -> bool:
    """
    Return whether ``data`` is a preprocessing handler class or instance.

    Args:
        data: Candidate runtime object.

    Returns:
        ``True`` for subclasses or instances of ``AbstractPreprocessingHandler``.
    """
    if inspect.isclass(data):
        return issubclass(data, AbstractPreprocessingHandler)

    return isinstance(data, AbstractPreprocessingHandler)


def is_pt_filepath(source: Any) -> bool:
    """
    Return whether ``source`` looks like a torch cache file path.

    Args:
        source: Candidate path string.

    Returns:
        ``True`` when ``source`` is a string ending with ``.pt``.
    """
    return isinstance(source, str) and source.endswith('.pt')


def is_pkl_filepath(source: Any) -> bool:
    """
    Return whether ``source`` looks like a pickle cache file path.

    Args:
        source: Candidate path string.

    Returns:
        ``True`` when ``source`` is a string ending with ``.pkl``.
    """
    return isinstance(source, str) and source.endswith('.pkl')
