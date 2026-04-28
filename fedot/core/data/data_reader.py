import os
from typing import Optional, Union, List

import numpy as np
from scipy.io.arff import loadarff

from fedot.core.data.data_reader_rules import resolve_arff_target_idx, split_arff_features_and_target

from fedot.core.backend.backend import Backend
from fedot.core.data.complex_types import PathType


def get_df_from_csv(
    file_path: PathType,
    delimiter: str,
    index_col: Optional[Union[str, int]] = None,
    possible_idx_keywords: Optional[List[str]] = None,
    *,
    columns_to_drop: Optional[List[Union[str, int]]] = None,
    columns_to_use: Optional[List[Union[str, int]]] = None,
    nrows: Optional[int] = None,
):
    """
    Load a CSV file into a dataframe with optional column selection and index handling.

    The function supports:
    - Automatic `index_col` detection when `index_col` is not provided.
    - Either `columns_to_drop` or `columns_to_use` (mutually exclusive).
    - Limiting the number of read rows via `nrows`.

    Args:
        file_path (PathType): Path to the input CSV file.
        delimiter (str): CSV delimiter (passed as `sep` to `read_csv`).
        index_col (Optional[Union[str, int]]): Column to use as an index.
            If `None`, the function may try to infer the index column using
            `possible_idx_keywords`.
        possible_idx_keywords (Optional[List[str]]): Keywords used to decide whether
            a column name is suitable for the index (case-insensitive substring match).
        columns_to_drop (Optional[List[Union[str, int]]]): Columns to exclude from the
            resulting dataframe. Cannot be used together with `columns_to_use`.
        columns_to_use (Optional[List[Union[str, int]]]): Columns to keep in the
            resulting dataframe. Cannot be used together with `columns_to_drop`.
        nrows (Optional[int]): Maximum number of rows to read.

    Returns:
        pandas.DataFrame: Loaded dataframe (CPU/GPU dependent on `backend.pd`).
    """
    pd_backend = Backend().pd

    def define_index_column(candidate_columns: List[str]) -> Optional[str]:
        for column_name in candidate_columns:
            if is_column_name_suitable_for_index(column_name):
                return column_name
        return None

    def is_column_name_suitable_for_index(column_name: str) -> bool:
        return any(key in column_name.lower() for key in possible_idx_keywords)

    columns_to_drop = columns_to_drop or []
    columns_to_use = columns_to_use or []
    possible_idx_keywords = possible_idx_keywords or []

    columns = pd_backend.read_csv(file_path, sep=delimiter, index_col=False, nrows=1).columns

    if columns_to_drop and columns_to_use:
        raise ValueError(
            "Incompatible arguments are used: columns_to_drop and columns_to_use. "
            "Only one of them can be specified simultaneously."
        )

    if columns_to_drop:
        columns_to_use = [col for col in columns if col not in columns_to_drop]
    elif not columns_to_use:
        columns_to_use = list(columns)

    candidate_idx_cols = [columns_to_use[0], columns[0]]
    if index_col is None:
        defined_index = define_index_column(candidate_idx_cols)
        if defined_index is not None:
            index_col = defined_index

    if (index_col is not None) and (index_col not in columns_to_use):
        columns_to_use.append(index_col)

    return pd_backend.read_csv(
        file_path,
        sep=delimiter,
        index_col=index_col,
        usecols=columns_to_use,
        nrows=nrows,
    )


def read_arff_file(file_path: PathType,
                   target_idx: Optional[Union[int, str]] = None):
    """
    Read an ARFF file and return `(features, target)` arrays.

    The function uses the current global `backend` to decide whether to return
    NumPy arrays (CPU) or CuPy arrays (GPU). Target column detection can be
    provided explicitly via `target_idx` or inferred automatically:
    - If `target_idx` is `None`, it tries to detect whether the last or the first
      column contains string values (byte strings after reading).
    - If `target_idx` is a `str`, it is treated as a target column name.

    Args:
        file_path (PathType): Path to the input `.arff` file.
        target_idx (Optional[Union[int, str]]): Target column index or name.
            If `None`, the target is inferred automatically. If a string is provided,
            it is resolved against `meta.names()`.

    Returns:
        Tuple[Any, Optional[Any]]:
            - features: array of predictors (shape depends on the dataset).
            - target: target array or `None` if no target column is detected.
    """
    xp = Backend().xp
    backend_name = Backend().name

    data, meta = loadarff(file_path)

    data_array = np.asarray([data[name] for name in meta.names()])

    target_resolution = resolve_arff_target_idx(
        target_idx=target_idx,
        field_names=meta.names(),
        data_array=data_array,
    )

    features, target = split_arff_features_and_target(
        data_array=data_array,
        target_idx=target_resolution.target_idx,
    )

    if backend_name == "gpu":
        features = xp.asarray(features)
        if target is not None:
            target = xp.asarray(target)

    return features, target
