from functools import reduce
from typing import Optional

import numpy as np


def find_common_elements(*arrays: np.array) -> np.array:
    """
    Intersects all of the ndarrays in ``arrays`` and returns their common elements

    :param arrays: tuple of ndarrays to be intersected

    :return: ndarray of unique elements common to all the provided ``arrays``
        or empty ndarray if none of them intersects
    """
    common_elements = reduce(np.intersect1d, arrays[1:], arrays[0])
    return common_elements


def flatten_extra_dim(data: Optional[np.array]) -> Optional[np.array]:
    """
    Removes last dimension if it is equal to one

    :param data: ndarray to be reshaped

    :return: reshaped view of the original array or None if input is None
    """

    if data is not None and data.shape[-1] == 1:
        return np.squeeze(data, axis=-1)
    return data


def atleast_n_dimensions(data: np.array, ndim: int) -> np.array:
    """
    Returns a view of the ``data` with at least ``ndim`` dimensions

    :param data: ndarray which dimensional size should be set to at least ``ndim``
    :param ndim: number of required axes to have in ``data``

    :return: ``data`` expanded from the last axis to the provided ``ndim`` size if it doesn't satisfy it
    """
    while data.ndim < ndim:
        data = np.expand_dims(data, axis=-1)
    return data


def atleast_2d(data: np.array) -> np.array:
    """
    Returns a view of the ``data` with at least `2` dimensions

    :param data: ndarray which dimensional size should be set to at least 2

    :return: ``data`` expanded from the last axis to `2` dimensions size if it doesn't satisfy it
    """
    return atleast_n_dimensions(data, ndim=2)


def atleast_4d(data: np.array) -> np.array:
    """
    Returns a view of the ``data` with at least `4` dimensions

    :param data: ndarray which dimensional size should be set to at least 4

    :return: ``data`` expanded from the last axis to `4` dimensions size if it doesn't satisfy it
    """
    return atleast_n_dimensions(data, ndim=4)
