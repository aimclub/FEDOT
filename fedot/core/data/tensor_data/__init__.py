from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.lazy_tensor import LazyTensor
from fedot.core.data.tensor_data.tensor_data import TensorData

__all__ = [
    'DataSpec',
    'LazyTensor',
    'TensorData',
    'TensorDataCreator',
]


def __getattr__(name):
    if name == 'TensorDataCreator':
        from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator

        return TensorDataCreator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
