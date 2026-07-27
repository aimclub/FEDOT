from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.io.arff import loadarff

from fedot.core.backend.backend import Backend
from fedot.core.data.common.types import PathType
from fedot.core.data.tensor_data.tools import convert_bytes


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

    columns = pd_backend.read_csv(
        file_path, sep=delimiter, index_col=False, nrows=1).columns

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


def read_arff_file(file_path: PathType) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load an ARFF file via SciPy and return the full attribute matrix plus field names.

    Each row of ``features`` corresponds to one ``@attribute`` in file order (same order
    as :meth:`scipy.io.arff.MetaData.names`). Columns are instances / data rows. Values
    are passed through :func:`~fedot.core.data.tensor_data.tools.convert_bytes` for
    byte-string decoding and numeric coercion where possible.

    When the active FEDOT backend is ``gpu``, the feature array is moved to the backend
    array module (CuPy) via ``Backend().xp``.

    Args:
        file_path: Path to the ``.arff`` file.

    Returns:
        A pair ``(features, field_names)`` where:

        * ``features`` — ``ndarray`` of shape ``(n_attributes, n_instances)``.
        * ``field_names`` — list of attribute names from the ARFF header, aligned with
          rows of ``features``; ``None`` if the header declares no attributes (empty
          name list).
    """
    xp = Backend().xp

    data, meta = loadarff(file_path)

    field_names = meta.names()
    if len(field_names) == 0:
        field_names = None

    data_array = np.asarray([data[name] for name in meta.names()])

    features = convert_bytes(data_array)

    if xp.__name__ == 'cupy':
        features = xp.asarray(features)

    return features, field_names
