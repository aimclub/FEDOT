from typing import Any
import numpy as np
from fedot.core.backend.backend import Backend


def to_numpy(array: Any) -> np.ndarray:
    """Convert arrays from the active backend to a NumPy array for byte hashing."""
    backend = Backend()

    if backend.device.type != "cpu":
        return backend.xp.asnumpy(array)
    return np.asarray(array)
