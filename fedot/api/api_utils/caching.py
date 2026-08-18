from fedot.core.caching.cacher import Cacher
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation import Operation


def try_load_data(data: TensorData, operation: Operation):
    cacher = Cacher()
    cached_data = cacher.load_tensor_data(input_data=data, operation=operation)
    return cached_data


def try_load_operation(operation: Operation, data: TensorData):
    cacher = Cacher()
    cached_operation = cacher.load_operation(operation, data)
    return cached_operation


def cache(data: TensorData, operation: Operation):
    cacher = Cacher()
    Cacher.cache_tensor_data()
    Cacher.cache_operation()