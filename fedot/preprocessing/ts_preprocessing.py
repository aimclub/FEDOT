# TODO romankuklo: move to obligatory ts preprocessing service
from typing import Optional, List

from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data.tools import get_idx_from_features_names
from fedot.core.data.common.types import ArrayType, IndexType

from fedot.core.data.common.enums import TSOrientationEnum, StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.tensor_data.tools import replace_missing_with_np_nan


def long_to_wide(features: ArrayType,
                 features_names: Optional[List[str]] = None,
                 terms_idx: IndexType = None):
    """
    Convert time series from `long` format to `wide` format.

    Expected `features` in `long` format is a 2D array where:
    - one column (`terms_idx`) stores a time/term label for each observation,
    - another column (`value_idx`) stores the observed value,
    - rows correspond to samples/observations that share the same labels.

    The function groups rows by unique labels in `terms_idx` and stacks the values
    so that the output has a wide layout where the term dimension becomes the second axis.

    Args:
        features (ArrayType): Input features in long format (typically 2D).
        features_names (Optional[List[str]]): Feature names, used when `terms_idx` is a name.
        terms_idx (IndexType): Index (or name) of the column that contains term labels.
            If `None`, the function uses `terms_idx = 0`.

    Returns:
        Tuple[ArrayType, ArrayType]: `(wide, unique_labels)` where:
            - wide is the converted 3D/stacked wide array (dtype float32),
            - unique_labels are the unique term labels from the input (dtype depends on backend).
    """
    xp = Backend().xp

    if terms_idx is None:
        terms_idx = 0
    else:
        terms_idx = get_idx_from_features_names(terms_idx, features_names)[0]

    value_idx = 1 - terms_idx

    unique_labels = xp.unique(features[:, terms_idx])

    split_arrays = [
        xp.asarray(
            features[features[:, terms_idx] == label, value_idx],
            dtype=xp.float32
        )
        for label in unique_labels
    ]

    lengths = [arr.shape[0] for arr in split_arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All series must have the same length to convert to wide format.")

    wide = xp.stack(split_arrays, axis=0).astype(xp.float32)
    return wide, unique_labels


def reshape_and_get_init_shape(features: ArrayType):
    """
    Normalize input time series with potentially multiple channels.

    The function ensures that multi-channel time series do not exceed 3 dimensions:
    - If `features.ndim == 1`, it is expanded to shape `(1, T)`.
    - If `features.ndim == 2`, it keeps `(B, T)` and returns `init_shape = None`.
    - If `features.ndim == 3`, it reshapes `(T, B, C)` into `(T, B * C)` and
      returns `init_shape = (T, B, C)` for potential later reshaping.

    Args:
        features (ArrayType): Time-series array.

    Returns:
        Tuple[ArrayType, Optional[tuple]]: `(features_normalized, init_shape)` where
            `init_shape` is `None` unless the input was 3D.
    """
    xp = Backend().xp

    if features.ndim == 1:
        features = xp.expand_dims(features, axis=0)
        init_shape = features.shape
    elif features.ndim == 2:
        init_shape = features.shape
    elif features.ndim == 3:
        init_shape = features.shape
        T, B, C = features.shape
        features = features.reshape(T, B * C)
    elif features.ndim > 3:
        raise ValueError("Multichannel time series must not have more than 3 dimensions")

    return features, init_shape


def process_ts_data(
    features: ArrayType,
    target: ArrayType = None,
    features_names: Optional[List[str]] = None,
    state: StateEnum = StateEnum.FIT,
    ts_orientation: Optional[TSOrientationEnum] = None,
    terms_idx: int = None,
    forecast_horizon: int = None,
    data_type: DataTypesEnum = DataTypesEnum.ts,
    without_target: bool = False
):
    """
    Apply time-series preprocessing and optional forecasting split.

    For non-time-series datasets (`data_type == DataTypesEnum.tabular`), the function
    returns inputs unchanged.

    For time-series datasets it:
    1. Normalizes multi-channel inputs via :func:`check_multichannel_ts`.
    2. Optionally converts from `long` to `wide` representation depending on
       `ts_orientation`.
    3. In `FIT` mode and when `forecast_horizon` is provided, splits the array into:
       - new `target` containing the last `forecast_horizon` steps,
       - new `features` containing the remaining prefix steps.

    Args:
        features (ArrayType): Input feature array (tabular or time-series).
        target (ArrayType): Optional target array.
        features_names (Optional[List[str]]): Feature names for resolving `terms_idx` by name.
        state (StateEnum): Pipeline state (`FIT` or `PREDICT`).
        ts_orientation (Optional[TSOrientationEnum]): Time-series orientation (`wide` or `long`).
        terms_idx (int): Column index (or name, depending on `get_idx_from_features_names`)
            that defines the term label in `long` format.
        forecast_horizon (int): If set and `state == StateEnum.FIT`, splits target/features
            using the last `forecast_horizon` time steps.
        data_type (DataTypesEnum): Dataset type.

    Returns:
        Tuple[ArrayType, ArrayType, Optional[tuple], int]:
            - processed_features (ArrayType)
            - processed_target (ArrayType or None)
            - init_shape (Optional[tuple]): original `(B, C, T)` when input was 3D.
            - terms_idx (int): resolved term index used for long->wide conversion.
    """
    if data_type == DataTypesEnum.tabular:
        return features, target, None, None

    features, init_shape = reshape_and_get_init_shape(features)

    features = replace_missing_with_np_nan(features)

    if isinstance(ts_orientation, str):
        ts_orientation = TSOrientationEnum(ts_orientation)

    if ts_orientation is None:
        ts_orientation = TSOrientationEnum.wide
    elif ts_orientation == TSOrientationEnum.long:
        features, terms_idx = long_to_wide(features, features_names, terms_idx)

    if without_target:
        return features, target, init_shape, terms_idx

    if state == StateEnum.FIT and forecast_horizon is not None:
        target = features[features.shape[1] - forecast_horizon:, :]
        features = features[:-forecast_horizon, :]

    return features, target, init_shape, terms_idx
