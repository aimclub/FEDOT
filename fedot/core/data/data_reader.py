import numpy as np
import pandas as pd
import os
from fedot.core.backend.backend import backend
from fedot.core.data.complex_types import PathType
from typing import Optional, Union, List
from fedot.core.data.data_tools import convert_bytes
from scipy.io.arff import loadarff


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
    pd_backend = backend.pd

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
    xp = backend.xp
    backend_name = backend.name

    data, meta = loadarff(file_path)

    data_array = np.asarray([data[name] for name in meta.names()])

    if target_idx is not None:
        if isinstance(target_idx, str):
            target_idx = meta.names().index(target_idx)
    else:
        if isinstance(convert_bytes(data_array[-1])[0], str):
            target_idx = -1
        elif isinstance(convert_bytes(data_array[0])[0], str):
            target_idx = 0
        else:
            target_idx = None

    if target_idx is not None:
        target = data_array[target_idx]
        features = np.delete(data_array, target_idx, axis=0)
        target = convert_bytes(target)
    else:
        target = None
        features = data_array

    features = convert_bytes(features)

    if backend_name == "gpu":
        features = xp.asarray(features)
        if target is not None:
            target = xp.asarray(target)

    return features, target
