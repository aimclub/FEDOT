import logging
import numbers
import os
from typing import Optional, Union, List, Tuple, Any, Dict

import numpy as np
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.common.types import PathType, PandasType, ArrayType, IndexType
from fedot.core.data.common.enums import StateEnum

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.tools.index_mapping_tools import update_index_mapping


logger = logging.getLogger(__name__)


def is_existed_csv_path(path: PathType) -> bool:
    """
    Check whether the provided path points to an existing CSV file.

    Args:
        path (PathType): Path to check.

    Returns:
        bool: True if the file exists and has a `.csv` extension, otherwise False.
    """
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False


def get_device_from_str(device: Union[str, torch.device]) -> torch.device:
    """
    Convert a device representation to `torch.device`.

    Args:
        device (Union[str, torch.device]): Either an already created `torch.device`
            or a string like `"cpu"` / `"cuda"`.

    Returns:
        torch.device: Parsed torch device.
    """
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def tensor_memory_usage(value: Any) -> int:
    """
    Estimate memory occupied by a torch tensor in bytes.

    Args:
        value: Value to inspect.

    Returns:
        int: Tensor payload size in bytes, or `0` for non-tensor values.
    """
    if isinstance(value, torch.Tensor):
        return value.element_size() * value.nelement()
    return 0


def convert_bytes(x: np.ndarray) -> np.ndarray:
    """
    Convert decoded byte/string data to numeric values when possible.

    The function attempts to:
    1) Decode UTF-8 byte strings into Python strings.
    2) Cast to float.
    3) If float conversion fails, cast to `str`.

    Args:
        x (np.ndarray): Input array that may contain byte strings.

    Returns:
        np.ndarray: Converted array with dtype float or str.
    """
    # Conversion of target values to float or str
    try:
        x = np.char.decode(x, encoding='utf-8')
    except BaseException:
        pass
    try:
        x = x.astype('float')
    except ValueError:
        x = x.astype(str)
    return x


def get_values_from_df(df: PandasType) -> ArrayType:
    """
    Extract values from a dataframe-like object.

    Tries `df.values` first. If it fails (e.g., for some cuDF cases),
    falls back to `df.to_pandas().values`.

    Args:
        df (PandasType): Input dataframe (CPU pandas or GPU cuDF).

    Returns:
        ArrayType: Underlying values as a numpy-like array.
    """
    try:
        features = df.values
        return features
    except Exception as e:
        features = df.to_pandas().values
        logger.info(
            f"Using pandas instead of cudf. Failed to get values from cudf DataFrame.")
        return features


def replace_missing_with_np_nan(arr: ArrayType) -> ArrayType:
    """
    Replace missing values with `NaN` and normalize dtype for CPU backends.

    For GPU backend (`backend.name == "gpu"`) this function returns `arr` unchanged.
    For CPU it attempts to:
    - Cast numeric dtypes to float32.
    - For object dtype, treat `None` and pandas/cudf missing values as `xp.nan`.
    - Convert non-missing values to float when possible; keep strings as-is.

    Args:
        arr (ArrayType): Input array.

    Returns:
        ArrayType: Array with missing values replaced by `NaN` (or `xp.nan`).
    """
    if arr is None:
        return arr

    xp = Backend().xp
    pd_backend = Backend().pd
    backend_name = Backend().name

    if backend_name == "gpu":
        return arr

    try:
        xp_arr = xp.asarray(arr)
        if xp_arr.dtype.kind in ("i", "u", "f", "b"):
            return xp_arr.astype(xp.float32, copy=False)
    except Exception:
        pass

    arr_obj = xp.asarray(arr, dtype=object)
    has_string = False

    missing_strings = {"", "nan", "none", "null", "na", "n/a"}

    def _normalize(x: object):
        nonlocal has_string

        if x is None or pd_backend.isna(x):
            return xp.nan

        if isinstance(x, str):
            stripped = x.strip().lower()
            if stripped in missing_strings:
                return xp.nan

            has_string = True
            return x

        try:
            return float(x)
        except (TypeError, ValueError):
            return xp.nan

    out = xp.vectorize(_normalize, otypes=[object])(arr_obj)

    if not has_string:
        return xp.asarray(out, dtype=xp.float32)

    return out


def _drop_rows_with_nan_in_target(features: ArrayType,
                                  target: ArrayType) -> Tuple[ArrayType, ArrayType]:
    """
    Remove samples whose target contains missing values.

    Missing target values are detected for numeric arrays with backend `isnan`
    logic and for object/string arrays with pandas-style NA checks plus common
    textual missing markers.

    Args:
        features (ArrayType): Feature matrix aligned with `target` by rows.
        target (ArrayType): Target array to inspect for missing values.

    Returns:
        Tuple[ArrayType, ArrayType]: Features and target filtered to rows with
            complete target values. If `target` is None, inputs are returned
            unchanged.

    Raises:
        ValueError: If all rows contain missing target values.
    """
    xp = Backend().xp
    pd_backend = Backend().pd

    if target is None:
        return features, target

    target = xp.asarray(target)

    if target.dtype.kind == "f":
        nan_mask = xp.isnan(target)
    elif target.dtype.kind in ("i", "u", "b"):
        nan_mask = xp.zeros(target.shape, dtype=bool)
    else:
        missing_strings = {"", "nan", "none", "null", "na", "n/a"}

        def _is_missing(x):
            if x is None:
                return True
            try:
                if pd_backend.isna(x):
                    return True
            except Exception:
                pass
            if isinstance(x, str) and x.strip().lower() in missing_strings:
                return True
            return False

        nan_mask = xp.vectorize(_is_missing, otypes=[bool])(target)

    number_nans_per_rows = nan_mask.sum(axis=1)
    non_nan_row_ids = xp.ravel(xp.argwhere(number_nans_per_rows == 0))

    if non_nan_row_ids.size == 0:
        raise ValueError("Data contains too much nans in the target column(s)")

    return features[non_nan_row_ids, :], target[non_nan_row_ids, :]


def atleast_n_dimensions(data: ArrayType, ndim: int) -> ArrayType:
    """
    Expand `data` until it has at least `ndim` dimensions.

    This is done by repeatedly adding axes at the end of the array shape using
    `xp.expand_dims`.

    Args:
        data (ArrayType): Input array.
        ndim (int): Minimum number of dimensions required.

    Returns:
        ArrayType: Expanded array view/copy with `data.ndim >= ndim`.
    """
    xp = Backend().xp

    while data.ndim < ndim:
        data = xp.expand_dims(data, axis=-1)
    return data


def convert_idx_to_list(idx: IndexType) -> IndexType:
    """
    Normalize an index-like value to list.

    Args:
        idx (IndexType): Index input to normalize.

    Returns:
        List: Normalized index representation.
    """

    if isinstance(idx, list) or idx is None:
        return idx
    if isinstance(idx, (int, str)):
        return [idx]
    if isinstance(idx, np.ndarray) or isinstance(idx, Backend().xp.ndarray):
        return idx.tolist()
    else:
        return list(idx)


def get_idx_from_features_names(idx: IndexType,
                                features_names: Optional[List[str]]) -> IndexType:
    """
    Convert feature names (strings) to feature indices.

    If `idx` already contains integers, it is returned unchanged.
    If `idx` contains strings, `features_names` must be provided and is used to map
    each name to its integer position.

    Args:
        idx (IndexType): Indices or feature names.
        features_names (Optional[List[str]]): List of all feature names in order.

    Returns:
        IndexType: Indices as an array/module-compatible representation.
    """

    if idx is None or len(idx) == 0:
        return idx

    first = idx[0]

    if isinstance(first, numbers.Integral):
        return idx

    if features_names is None:
        raise ValueError(
            "Impossible to specify categorical features by name when features_names are not specified"
        )

    try:
        if isinstance(first, str):
            name_to_index = {name: i for i, name in enumerate(features_names)}
            return [name_to_index[name] for name in idx]

    except KeyError as e:
        raise ValueError(
            f"Feature name '{e.args[0]}' was not found in features_names: {features_names}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to get index from feature names: {idx}") from e

    raise TypeError(
        f"idx must contain either integers or strings, got elements of type {type(first).__name__}"
    )


def get_target_and_features(
    features: ArrayType,
    target: ArrayType,
    features_names: Optional[List[str]] = None,
    target_idx: IndexType = None,
    state: StateEnum = StateEnum.FIT,
    data_type: DataTypesEnum = DataTypesEnum.tabular,
    idx_mapping: Optional[Dict[int, int]] = None,
    without_target: bool = False
) -> Tuple[ArrayType, ArrayType, Any]:
    """
    Split and preprocess `(features, target)` from raw array-like inputs.

    In `FIT` mode (when `state == StateEnum.FIT`) this function:
    - Determines the target either from `target` argument or from `target_idx` (or defaults
      to the last feature column),
    - Removes the target column(s) from `features`,
    - Replaces missing values in `target` with `NaN`,
    - Encodes the target via `encode_target`,
    - Drops rows where the target contains NaNs.

    In non-`FIT` mode it returns `(features, None, None)`.

    Args:
        features (ArrayType): Feature matrix `(n_samples, n_features)`.
        target (ArrayType): Optional target array. If provided, it is used as-is.
        features_names (Optional[List[str]]): Feature names for name-to-index conversion.
        target_idx (IndexType): Target column index or name (optional).
        state (StateEnum): Fit or transform mode.
        data_type (DataTypesEnum): Dataset type. For `DataTypesEnum.ts` the function returns
            inputs unchanged except that `target_encoder` is `None`.

    Returns:
        Tuple[ArrayType, ArrayType, Any]:
            - features_processed (ArrayType)
            - target_processed (ArrayType) or `None`
            - target_encoder (Any) or `None`
    """
    # For ts type if target_idx is not provided, do nothing
    if data_type == DataTypesEnum.ts and target_idx is None:
        return features, target, idx_mapping

    # replace missing values with np.nan for ts is already done in ts preprocessing
    if data_type != DataTypesEnum.ts:
        features = replace_missing_with_np_nan(features)

    if without_target:
        return features, target, idx_mapping

    xp = Backend().xp

    if state == StateEnum.FIT:
        if target is not None:
            target = xp.array(target)
        else:
            if target_idx is not None:
                target_idx = get_idx_from_features_names(
                    target_idx, features_names)
                target = features[:, target_idx].copy()
            else:
                if data_type == DataTypesEnum.ts:
                    raise ValueError("Target is not provided and target_idx is not provided."
                                     "Change your task, or provide target or target_idx explicitly.")
                target = features[:, -1].copy()
                target_idx = [-1]

            features = xp.delete(features, target_idx, axis=1)

            idx_mapping = update_index_mapping(
                idx_mapping, target_idx, features)

        target = atleast_n_dimensions(target, 2)
        target = replace_missing_with_np_nan(target)
        try:
            target = xp.asarray(target, dtype=xp.float32)
        except BaseException:
            pass

        features, target = _drop_rows_with_nan_in_target(features, target)

        return features, target, idx_mapping

    return features, None, idx_mapping


def transform_to_tensor(features: ArrayType,
                        target: ArrayType,
                        ts_init_shape: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert features and target arrays to torch tensors (float32 by default).

    If `text_idx` is provided, the corresponding columns are removed from `features`.
    When `features` becomes empty, it uses `text_tensors` as the features tensor.
    Optionally reshapes features to `ts_init_shape` (useful for time series layouts).

    Args:
        features (ArrayType): Input feature matrix.
        target (ArrayType): Target array.
        text_tensors (Optional[torch.Tensor]): Precomputed text embeddings to append
            when `text_idx` columns are removed.
        text_idx (Optional[IndexType]): Indices of text columns to remove from `features`.
        ts_init_shape (Any): Optional target shape for reshaping the features tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: `(features_tensor, target_tensor)`.
    """

    features = to_tensor(features, dtype=torch.float32)

    if ts_init_shape is not None and len(ts_init_shape) == 3:
        features = features.reshape(ts_init_shape)

    target = to_tensor(target, dtype=torch.float32)

    return features, target


def delete_zero_features(features: ArrayType) -> ArrayType:
    xp = Backend().xp

    non_zero_mask = xp.any(features != 0, axis=0)

    return features[:, non_zero_mask]


def to_tensor(array: ArrayType, dtype=None) -> torch.Tensor:
    """
    Convert an array-like object to a torch tensor on the active backend device.

    The function casts the input to `xp.float64` before creating the tensor, and
    places it on `backend.device`.

    Args:
        array (ArrayType): Input array. If `None`, returns `None`.
        dtype: Optional torch dtype passed to `torch.tensor`.

    Returns:
        torch.Tensor: Tensor placed on `backend.device` (or `None` if `array is None`).
    """
    xp = Backend().xp
    device = Backend().device

    if array is None:
        return None

    array = array.astype(xp.float64)
    return torch.tensor(array, dtype=dtype, device=device)
