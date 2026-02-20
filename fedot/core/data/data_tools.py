import os
import pandas as pd
import numpy as np
from typing import Optional, Union
from fedot.core.data.data_preprocessing import find_categorical_columns


def replace_missing_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all missing values (None, np.nan, pd.NA, etc.) in a pandas DataFrame with float('nan').

    Args:
        df: Input pandas DataFrame with missing values.

    Returns:
        DataFrame with all missing values replaced by float('nan').
    """
    # Replace all missing values with np.nan (which is equivalent to float('nan'))
    df_replaced = df.copy()
    df_replaced = df_replaced.apply(
        lambda x: x.fillna(np.nan) if x.dtype.kind in ['i', 'f', 'O'] else x
    )

    # Ensure all None values are replaced (for object columns)
    df_replaced = df_replaced.where(df_replaced.notna(), np.nan)

    return df_replaced


def _drop_rows_with_nan_in_target(features: np.ndarray, 
                                  target: np.ndarray) -> pd.DataFrame:
    """
    Drops rows with nans in target column

    Args:
        data: to be modified

    Returns:
        modified ``data``
    """
    if target is None:
        return features, target
    # Find indices of nans rows. Using pd instead of np because it is needed for string columns
    bool_target = np.array(pd.isna(target))
    number_nans_per_rows = bool_target.sum(axis=1)

    # Ids of rows which doesn't contain nans in target
    non_nan_row_ids = np.ravel(np.argwhere(number_nans_per_rows == 0))

    if len(non_nan_row_ids) == 0:
        raise ValueError('Data contains too much nans in the target column(s)')

    features = features[non_nan_row_ids, :]
    target = target[non_nan_row_ids, :]

    return features, target


def encode_categorical_features(array: Union[pd.DataFrame, np.ndarray], 
                               categorical_idx: Optional[np.ndarray] = None) -> np.ndarray:
    if categorical_idx is None:
        categorical_idx, _ = find_categorical_columns(array)
     # If no categorical columns, return original array
    if len(categorical_idx) == 0:
        return array, None

    # Encode each categorical column
    for idx in categorical_idx:
        column = array[:, idx]
        _, codes = np.unique(column, return_inverse=True)
        codes = codes.astype(float)
        codes[codes == -1] = np.nan
        array[:, idx] = codes

    return array, categorical_idx


def encode_target(target: np.ndarray) -> np.ndarray:
    """
    Encodes categorical target values using factorization.

    Args:
        target: Input 2D numpy array of shape [N, 1] with categorical target values.

    Returns:
        Encoded target array of the same shape [N, 1].
    """
    if target is None or len(target) == 0 or not isinstance(target[0, 0], str):
        return target
    
    target_flat = target.flatten()
    _, codes = np.unique(target_flat, return_inverse=True)
    codes = codes.astype(float)
    codes[codes == -1] = np.nan
    encoded_target = codes.reshape(-1, 1)
    return encoded_target


def is_existed_csv_path(path: str) -> bool:
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False

