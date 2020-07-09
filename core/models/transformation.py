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
            ((a.shape[-2] - shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - shape[-1]) // dx + 1,) + \
            shape
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]

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
    features = array.shape[1]
    res3d = _roll(array, (window_len, features)).reshape(-1, window_len, features)
    return res3d


def ts_to_lagged_3d(input_data: InputData) -> InputData:
    window_len, prediction_len = extract_task_param(input_data.task)

    features = np.c_[input_data.features, input_data.target]
    features_sliding = _ts_to_3d(features[:-prediction_len], window_len)
    target_sliding = _ts_to_3d(input_data.target[prediction_len:], window_len)

    transformed_data = copy(input_data)
    transformed_data.idx = np.arange(target_sliding.shape[0])
    transformed_data.features = features_sliding
    transformed_data.target = target_sliding

    transformed_data.data_type = DataTypesEnum.ts_lagged_3d
    return transformed_data


def ts_to_lagged_table(input_data: InputData) -> InputData:
    _, prediction_len = extract_task_param(input_data.task)

    transformed_data = ts_to_lagged_3d(input_data)

    transformed_data.features = transformed_data.features.reshape(
        transformed_data.features.shape[0], -1)

    target_shape = input_data.target.shape[-1] if input_data.target.ndim > 1 else 1
    # take last prediction_len features
    transformed_data.target = transformed_data.target[:, -prediction_len:] \
        .reshape(-1, prediction_len * target_shape)

    transformed_data.data_type = DataTypesEnum.ts_lagged_table
    return transformed_data


def ts_lagged_table_to_3d(input_data: InputData) -> InputData:
    window_len, prediction_len = extract_task_param(input_data.task)

    transformed_data = copy(input_data)
    features_shape = transformed_data.features.shape[-1] // window_len
    transformed_data.features = transformed_data.features.reshape(
        -1, window_len, features_shape)

    target_shape = transformed_data.target.shape[-1] // prediction_len
    target = transformed_data.target.reshape(
        -1, prediction_len, target_shape)
    transformed_data.target = np.concatenate([transformed_data.features[:, prediction_len:, -target_shape:],
                                              target],
                                             axis=1)

    transformed_data.data_type = DataTypesEnum.ts_lagged_3d
    return transformed_data


def ts_lagged_3d_to_ts(input_data: InputData) -> InputData:
    # make inverse transformation
    _, prediction_len = extract_task_param(input_data.task)

    # we have some information lost on last prediction_len values
    transformed_data = copy(input_data)
    target_shape = transformed_data.target.shape[-1]
    num_features = transformed_data.features.shape[-1] - target_shape
    transformed_data.features = np.r_[input_data.features[:, 0, :-target_shape],
                                      input_data.features[-1, 1:, :-target_shape],
                                      np.zeros((prediction_len, num_features))]
    transformed_data.target = np.r_[
        input_data.features[:prediction_len, 0, -target_shape:],
        input_data.target[0, :-1],
        input_data.target[:, -1]
    ].squeeze()

    transformed_data.data_type = DataTypesEnum.ts
    return transformed_data


def ts_lagged_to_ts(input_data: InputData) -> InputData:
    transformed = ts_lagged_table_to_3d(input_data)
    res = ts_lagged_3d_to_ts(transformed)
    res.data_type = DataTypesEnum.ts
    return res


def ts_lagged_3d_to_lagged_table(input_data: InputData) -> InputData:
    _, prediction_len = extract_task_param(input_data.task)

    transformed_data = copy(input_data)

    transformed_data.features = transformed_data.features.reshape(
        transformed_data.features.shape[0], -1)
    # return forecast values only
    transformed_data.target = transformed_data.target[:, -prediction_len:] \
        .reshape(-1, transformed_data.target.shape[-1] * prediction_len)
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
    (DataTypesEnum.ts_lagged_table, DataTypesEnum.ts):
        ts_lagged_to_ts,
    (DataTypesEnum.ts_lagged_table, DataTypesEnum.ts_lagged_3d):
        ts_lagged_table_to_3d,
    (DataTypesEnum.ts_lagged_3d, DataTypesEnum.ts):
        ts_lagged_3d_to_ts,
    (DataTypesEnum.ts_lagged_3d, DataTypesEnum.ts_lagged_table):
        ts_lagged_3d_to_lagged_table
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
