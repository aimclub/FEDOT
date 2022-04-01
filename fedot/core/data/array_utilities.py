from functools import reduce
from typing import Optional

import numpy as np


def find_common_elements(*indices: np.array) -> np.array:
    """ Returns array with unique elements common to *all* indices
    or the first index if it's the only one. """
    common_elements = reduce(np.intersect1d, indices[1:], indices[0])
    return common_elements


def flatten_extra_dim(data: Optional[np.array]) -> Optional[np.array]:
    """ Removes extra dimension if it is equal to one.

    It's different from np.squeeze in that it operates only on the last axis.

    :return: reshaped view of the original array or None if input is None. """

    if data is not None and data.shape[-1] == 1:
        return data.reshape(data.shape[:-1])
    return data


def atleast_n_dimensions(data: np.array, ndim: int) -> np.array:
    """ Return a view with extra dimensions to the array if necessary,
    such that the result has the required number of dimensions."""
    while data.ndim < ndim:
        data = np.expand_dims(data, axis=-1)
    return data


def atleast_2d(data: np.array) -> np.array:
    return atleast_n_dimensions(data, ndim=2)


def atleast_4d(data: np.array) -> np.array:
    return atleast_n_dimensions(data, ndim=4)
