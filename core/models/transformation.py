from copy import copy
from typing import List

import numpy as np
import pandas as pd

from core.models.data import (
    InputData,
)
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import extract_task_param


# Rolling 2D window for ND array
def _roll(a,  # ND array
          shape,  # rolling 2D window array
          dx=1,  # horizontal step, abscissa, number of columns
          dy=1):  # vertical step, ordinate, number of rows

    shape = a.shape[:-2] + \
            ((a.shape[-2]) // dy,) + \
            ((a.shape[-1] - shape[-1]) // dx + 1,) + \
            shape

    strides = (a.strides[:-2] +
               (a.strides[-2] * dy,) +
               (a.strides[-1] * dx,) +
               a.strides[-2:])

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _ts_to_3d(array, window_len: int):
    """
    Makes 3d dataset of sliding windows with shape (n-window_len+1, window_len, features)
    from (n, features) array.
    array: np.ndarray or pd.DataFrame
    """
    if isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    if array.ndim == 1:
        # add aditional dim for 1d target
        array = array[:, None]
    features_sets_num = array.shape[1]
    res3d = _roll(array, (window_len, features_sets_num)). \
        reshape(-1, window_len, features_sets_num)
    return res3d


def ts_to_lagged_3d(input_data: InputData) -> InputData:
    window_len, prediction_len = extract_task_param(input_data.task)

    transformed_data = copy(input_data)

    features = input_data.features

    expected_len = len(input_data.idx)
    if input_data.task.task_params.make_future_prediction:
        expected_len = expected_len + 1

    if input_data.target is not None:
        target = copy(input_data.target)

        if (not np.array_equal(input_data.features, input_data.target) and
                input_data.features is not None):
            features = np.c_[input_data.features, input_data.target]
        else:
            # if no real exog variables
            features = input_data.target

        # fake target values for last predictions
        target = np.insert(target, len(target), [None] * prediction_len)

        target_sliding = _ts_to_3d(target, prediction_len)
        target_sliding = target_sliding[:expected_len, :, :]
        transformed_data.target = target_sliding
    else:
        transformed_data.target = None

    # fake features values for the first predictions
    if len(features.shape) == 1:
        features = np.insert(features, 0, [None] * (window_len))
    elif len(features.shape) == 2:
        new_features = []
        for features_dim in range(features.shape[1]):
            modified_feature = np.insert(features[:, features_dim], 0, [None] * (window_len))
            new_features.append(modified_feature)
        features = np.stack(np.asarray(new_features)).T
    else:
        raise NotImplementedError()

    features_sliding = _ts_to_3d(features, window_len)

    features_sliding = features_sliding[:expected_len, :, :]

    transformed_data.idx = np.arange(features_sliding.shape[0])
    transformed_data.features = features_sliding

    transformed_data.data_type = DataTypesEnum.ts_lagged_3d
    return transformed_data


def ts_to_lagged_table(input_data: InputData) -> InputData:
    _, prediction_len = extract_task_param(input_data.task)

    transformed_data = ts_to_lagged_3d(input_data)

    transformed_data.features = transformed_data.features.reshape(
        transformed_data.features.shape[0], -1)

    if input_data.target is not None:
        target_shape = input_data.target.shape[-1] if input_data.target.ndim > 1 else 1
        # take last prediction_len features
        transformed_data.target = transformed_data.target \
            .reshape(-1, prediction_len * target_shape)

    transformed_data.data_type = DataTypesEnum.ts_lagged_table
    return transformed_data


def direct(input_data: InputData) -> InputData:
    return copy(input_data)


# from datatype / to datatype : function
_transformation_functions_for_data_types = {
    (DataTypesEnum.ts, DataTypesEnum.ts_lagged_3d):
        ts_to_lagged_3d,
    (DataTypesEnum.ts, DataTypesEnum.ts_lagged_table):
        ts_to_lagged_table,
    (DataTypesEnum.forecasted_ts, DataTypesEnum.table): direct,
    (DataTypesEnum.table, DataTypesEnum.ts): direct

}


def transformation_function_for_data(input_data_type: DataTypesEnum,
                                     required_data_types: List[DataTypesEnum]):
    if input_data_type in required_data_types:
        return direct

    transformation = None
    for required_data_type in required_data_types:
        transformation = _transformation_functions_for_data_types.get(
            (input_data_type, required_data_type), None)
        if transformation:
            break

    if not transformation:
        raise ValueError(
            f'The {input_data_type} cannot be converted to {required_data_types}')
    return transformation
