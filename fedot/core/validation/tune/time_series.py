import numpy as np


def _calculate_n_splits(data, horizon: int):
    """ Calculated number of splits which will not lead to the errors """

    n_splits = len(data.features) // horizon
    # Remove one split to allow algorithm get more data for train
    n_splits = n_splits - 1
    return n_splits


def _choose_several_folds(n_splits):
    """ Choose ids of several folds for further testing """

    # If there not enough folds in time series - take all of them
    if n_splits < 3:
        return np.arange(0, n_splits)
    else:
        # Choose last folds for validation
        biggest_part = n_splits - 1
        medium_part = biggest_part - 1
        smallest_part = medium_part - 1
        return np.array([smallest_part, medium_part, biggest_part])
