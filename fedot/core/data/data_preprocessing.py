from typing import Optional, Tuple

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, data_type_is_multi_ts, data_type_is_table, data_type_is_ts
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID


def data_type_is_suitable_for_preprocessing(data: InputData) -> bool:
    return data_type_is_table(data) or data_type_is_ts(data) or data_type_is_multi_ts(data)


def replace_inf_with_nans(input_data: InputData):
    features = input_data.features
    is_inf = np.isin(features, [np.inf, -np.inf])
    if np.any(is_inf):
        features[is_inf] = np.nan


def replace_nans_with_empty_strings(input_data: InputData):
    """
    Replace NaNs with empty strings in input_data.features
    """
    input_data.features[pd.isna(input_data.features)] = ''


def convert_into_column(array: np.ndarray) -> np.ndarray:
    """ Perform conversion for data if it is necessary """
    if len(array.shape) == 1:
        return array.reshape(-1, 1)
    return array


def divide_data_categorical_numerical(input_data: InputData, categorical_ids: list,
                                      non_categorical_ids: list) -> Tuple[Optional[InputData], Optional[InputData]]:
    """
    Split tabular InputData into two parts: with numerical and categorical features
    using list with ids of categorical and numerical features.
    """
    numerical_input = input_data.subset_features(non_categorical_ids)
    categorical_input = input_data.subset_features(categorical_ids)
    if numerical_input or categorical_input:
        return numerical_input, categorical_input
    else:
        prefix = 'InputData contains no categorical and no numerical features.'
        raise ValueError(f'{prefix} Check data for Nans and inf values')


def find_categorical_columns(table: np.ndarray, column_type_ids: Optional[np.ndarray] = None):
    """
    Method for finding categorical and non-categorical columns in tabular data

    Args:
        table: tabular data for string columns types determination.
        column_type_ids: list with column type ids. If None, perform default checking.
    Returns:
        categorical_ids: indices of categorical columns in table.
        non_categorical_ids: indices of non categorical columns in table.
    """
    if column_type_ids is None:
        # Define if data contains string columns for "unknown table"
        return force_categorical_determination(table)

    is_str = column_type_ids == TYPE_TO_ID[str]
    categorical_ids = np.flatnonzero(is_str).tolist()
    non_categorical_ids = np.flatnonzero(~is_str).tolist()

    return categorical_ids, non_categorical_ids


def force_categorical_determination(table: np.ndarray):
    """ Find string columns using 'computationally expensive' approach """
    categorical_ids = []
    non_categorical_ids = []
    # For every column in table make check
    for column_id, column in enumerate(table.T):
        # Check if column is of string objects
        if pd.api.types.infer_dtype(column, skipna=True) == 'string':
            categorical_ids.append(column_id)
        else:
            non_categorical_ids.append(column_id)

    return categorical_ids, non_categorical_ids


def data_has_missing_values(data: InputData) -> bool:
    """ Check data for missing values."""
    return data_type_is_suitable_for_preprocessing(data) and pd.DataFrame(data.features).isna().to_numpy().sum() > 0


def data_has_categorical_features(data: InputData) -> bool:
    """
    Check data for categorical columns.
    Return bool, whether data has categorical columns or not
    """
    if data.data_type is not DataTypesEnum.table:
        return False

    feature_type_ids = data.supplementary_data.col_type_ids['features']
    cat_ids, non_cat_ids = find_categorical_columns(data.features, feature_type_ids)

    data.numerical_idx = non_cat_ids
    data.categorical_idx = cat_ids

    if len(cat_ids) > 0:
        data.categorical_features = data.subset_features(cat_ids).features

    return bool(cat_ids)


def data_has_text_features(data: InputData) -> bool:
    """
    Checks data for text fields.
    Data with text fields is always 1-dimensional due to the previous
    parsing of it from a general table to the distinct text data source.
    Returns bool, whether data has text fields or not
    """
    # TODO andreygetmanov: make compatible with current text checking
    return data.data_type is DataTypesEnum.text
