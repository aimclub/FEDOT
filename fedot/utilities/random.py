from typing import Optional, Union

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation, ModelImplementation
from fedot.core.utils import RandomStateHandler


class ImplementationRandomStateHandler(RandomStateHandler):
    def __init__(self, seed: Optional[int] = None,
                 implementation: Union[DataOperationImplementation, ModelImplementation] = None):
        if seed is None:
            super().__init__()
        else:
            super().__init__(seed)
        self.implementation = implementation
        self.implementation_old_random_state = None

    def __enter__(self):
        self.implementation_old_random_state = _set_operation_random_seed(self.implementation, self._seed)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _set_operation_random_seed(self.implementation, self.implementation_old_random_state)
        return super().__exit__(exc_type, exc_value, exc_traceback)


def _set_operation_random_seed(operation: Union[DataOperationImplementation, ModelImplementation],
                               seed: Optional[int]):
    old_random_state = None
    if hasattr(operation, 'random_state'):
        old_random_state = getattr(operation, 'random_state')
        setattr(operation, 'random_state', seed)

    elif hasattr(operation, 'seed'):
        old_random_state = getattr(operation, 'seed')
        setattr(operation, 'seed', seed)

    return old_random_state
