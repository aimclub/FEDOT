from typing import Tuple, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, data_type_is_table, data_type_is_ts, data_type_is_multi_ts
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID


def data_type_is_suitable_for_preprocessing(data: InputData) -> bool:
    return data_type_is_table(data) or data_type_is_ts(data) or data_type_is_multi_ts(data)


def replace_inf_with_nans(input_data: InputData):
    features = input_data.features
    has_infs = (features == np.inf) | (features == -np.inf)
    if np.any(has_infs):
        features[has_infs] = np.nan


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

    if len(categorical_ids) > 0 and len(non_categorical_ids) > 0:
        # Both categorical and numerical features
        numerical_input = input_data.subset_features(non_categorical_ids)
        categorical_input = input_data.subset_features(categorical_ids)
        return numerical_input, categorical_input

    elif len(categorical_ids) == 0 and len(non_categorical_ids) > 0:
        # Only numerical
        numerical_input = input_data.subset_features(non_categorical_ids)
        return numerical_input, None

    elif len(categorical_ids) > 0 and len(non_categorical_ids) == 0:
        # Only categorical
        categorical_input = input_data.subset_features(categorical_ids)
        return None, categorical_input

    else:
        prefix = 'InputData contains no categorical and no numerical features.'
        raise ValueError(f'{prefix} Check data for Nans and inf values')


def find_categorical_columns(table: np.ndarray, column_types: dict = None):
    """
    Method for finding categorical and non-categorical columns in tabular data

    :param table: tabular data for string columns types determination
    :param column_types: list with column types. If None, perform default checking
    :return categorical_ids: indices of categorical columns in table
    :return non_categorical_ids: indices of non categorical columns in table
    """
    if column_types is None:
        # Define if data contains string columns for "unknown table"
        return force_categorical_determination(table)

    categorical_ids = []
    non_categorical_ids = []
    for col_id, col_type_id in enumerate(column_types):
        if col_type_id == TYPE_TO_ID[str]:
            categorical_ids.append(col_id)
        else:
            non_categorical_ids.append(col_id)

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

    features_types = data.supplementary_data.column_types.get('features')
    cat_ids, non_cat_ids = find_categorical_columns(data.features, features_types)
    data_has_categorical_columns = len(cat_ids) > 0

    data.numerical_idx = non_cat_ids
    data.categorical_idx = cat_ids
    data.categorical_features = data.subset_features(cat_ids).features

    return data_has_categorical_columns


def data_has_text_features(data: InputData) -> bool:
    """
    Checks data for text fields.
    Data with text fields is always 1-dimensional due to the previous
    parsing of it from a general table to the distinct text data source.
    Returns bool, whether data has text fields or not
    """
    # TODO andreygetmanov: make compatible with current text checking
    return data.data_type is DataTypesEnum.text
