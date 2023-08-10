import random
from typing import Optional, Union

import numpy as np
from golem.core.utilities.random import RandomStateHandler

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation, ModelImplementation


class ImplementationRandomStateHandler(RandomStateHandler):
    MODEL_FITTING_SEED = 0

    def __init__(self, seed: Optional[int] = None,
                 implementation: Union[DataOperationImplementation, ModelImplementation] = None):
        super().__init__(seed)
        self.implementation = implementation

    def __enter__(self):
        _set_operation_random_seed(self.implementation, self._seed)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self._old_np_state)
        random.setstate(self._old_state)


def _set_operation_random_seed(operation: Union[DataOperationImplementation, ModelImplementation],
                               seed: Optional[int]):
    if hasattr(operation, 'random_state'):
        setattr(operation, 'random_state', seed)
    elif hasattr(operation, 'seed'):
        setattr(operation, 'seed', seed)
