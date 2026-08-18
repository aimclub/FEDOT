from fedot.core.caching.cacher import ensure_cacher
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation import Operation


def try_load_data(data: TensorData, operation: Operation):
    cacher = ensure_cacher(operation.cacher)
    cached_data = cacher.load_tensor_data(input_data=data, operation=operation)
    return cached_data


def try_load_operation(operation: Operation, data: TensorData):
    cacher = ensure_cacher(operation.cacher)
    cached_operation = cacher.load_operation(operation, data)
    return cached_operation


def cache(data: TensorData, operation: Operation):
    cacher = ensure_cacher(operation.cacher)
    cacher.cache_tensor_data(output_data=data, input_data=data, operation=operation)
    cacher.cache_operation(operation, data)
