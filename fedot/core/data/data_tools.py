import logging
import numbers
import os
from typing import Optional, Union, List, Tuple, Dict, Iterable, Any

import numpy as np
import torch

from fedot.core.backend.backend import backend
from fedot.core.data.complex_types import PathType, PandasType, ArrayType, IndexType
from fedot.core.data.tools import StateEnum

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.preprocessor_mapping import EMBEDDING_METHOD_MAPPING, ENCODER_MAPPING

from fedot.preprocessing.preprocessor_types import (EmbedderParameters, EmbeddingMethodEnum,
                                                    EncodingStrategyType, CategoricalEncodingDecision,
                                                    EncodingStrategyEnum)


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
        logger.info(f"Using pandas instead of cudf. Failed to get values from cudf DataFrame.")
        return features


def replace_missing_with_nan(arr: ArrayType) -> ArrayType:
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
    xp = backend.xp
    pd_backend = backend.pd
    backend_name = backend.name

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

    def _normalize(x: object):
        nonlocal has_string

        if x is None or pd_backend.isna(x):
            return xp.nan

        if isinstance(x, str):
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
    Drop rows where `target` contains NaN in any target column.

    Works for both numpy (CPU) and cupy (GPU) backends.

    Args:
        features (ArrayType): Feature matrix with shape `(n_samples, n_features)`.
        target (ArrayType): Target matrix with shape `(n_samples, n_targets)` or
            a compatible 2D representation.

    Returns:
        Tuple[ArrayType, ArrayType]: `(features_filtered, target_filtered)` with
            rows containing NaNs removed. If `target is None`, returns the inputs unchanged.
    """
    xp = backend.xp

    if target is None:
        return features, target

    target = xp.asarray(target)

    nan_mask = xp.isnan(target)
    number_nans_per_rows = nan_mask.sum(axis=1)
    non_nan_row_ids = xp.ravel(xp.argwhere(number_nans_per_rows == 0))

    if non_nan_row_ids.size == 0:
        raise ValueError("Data contains too much nans in the target column(s)")

    features = features[non_nan_row_ids, :]
    target = target[non_nan_row_ids, :]

    return features, target


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
    xp = backend.xp

    while data.ndim < ndim:
        data = xp.expand_dims(data, axis=-1)
    return data


def convert_idx_to_array(idx: IndexType) -> IndexType:
    """
    Normalize an index-like value to a backend array (NumPy/CuPy) when possible.

    Rules:
    - If `idx` is already a backend array (or `None`), it is returned as-is.
    - If `idx` is an `int` or `str`, it becomes a 1D array: `xp.array([idx])`.
    - Otherwise it is converted via `xp.array(idx)`.
    - On conversion error it falls back to returning Python lists.

    Args:
        idx (IndexType): Index input to normalize.

    Returns:
        IndexType: Normalized index representation.
    """
    xp = backend.xp

    if isinstance(idx, xp.ndarray) or idx is None:
        return idx
    try:
        if isinstance(idx, int) or isinstance(idx, str):
            return xp.array([idx])
        else:
            return xp.array(idx)
    except Exception:
        if isinstance(idx, (int, str)):
            return [idx]

        if isinstance(idx, Iterable):
            return list(idx)


def convert_to_list(idx: IndexType) -> List:
    """
    Convert index-like input to a Python `list` when appropriate.

    Args:
        idx (IndexType): Value to convert.

    Returns:
        List: Converted list. If `idx` is already a list or `None`, returns it unchanged.
            If `idx` is a numpy/CuPy array, returns `idx.tolist()`. Otherwise wraps
            the value into a single-element list.
    """
    if isinstance(idx, list) or idx is None:
        return idx
    elif isinstance(idx, np.ndarray) or isinstance(idx, backend.xp.ndarray):
        return idx.tolist()
    else:
        return [idx]


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
    xp = backend.xp

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
            return xp.array([name_to_index[name] for name in idx])

    except KeyError as e:
        raise ValueError(
            f"Feature name '{e.args[0]}' was not found in features_names: {features_names}"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to get index from feature names: {idx}") from e

    raise TypeError(
        f"idx must contain either integers or strings, got elements of type {type(first).__name__}"
    )


def get_embedder_parameters(parameters: Union[EmbedderParameters, Dict]) -> EmbedderParameters:
    """
    Normalize embedder configuration into an `EmbedderParameters` instance.

    Args:
        parameters (Union[EmbedderParameters, Dict]): Either an already created
            `EmbedderParameters` object or a dictionary with keys:
            `model_name`, `batch_size`, `device`, `method`.

            If an empty dict is provided, default parameters are used:
            model=`all-distilroberta-v1`, batch_size=`32`, device determined by CUDA availability,
            method=`transformer`.

    Returns:
        EmbedderParameters: Parsed/constructed embedder parameters.
    """
    if isinstance(parameters, Dict):
        if not parameters:
            parameters = EmbedderParameters(
                model_name='all-distilroberta-v1',
                batch_size=32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                method=EmbeddingMethodEnum.transformer
            )

        else:
            try:
                parameters = EmbedderParameters(model_name=parameters['model_name'],
                                                batch_size=parameters['batch_size'],
                                                device=torch.device(parameters['device']),
                                                method=EmbeddingMethodEnum(parameters['method']))
                return parameters
            except Exception as e:
                raise ValueError(f"Failed to get embedder parameters: {parameters}") from e

    elif isinstance(parameters, EmbedderParameters):
        return parameters

    else:
        raise ValueError(f"Invalid embedderparameters type: {type(parameters)}")


def get_text_embeddings(features: ArrayType,
                        text_idx: IndexType,
                        strategy: Union[EmbedderParameters, Dict],
                        features_names: Optional[List[str]] = None):
    """
    Compute text embeddings for selected text feature columns and zero-out them in `features`.

    If `text_idx` is `None`, returns `(None, None, features)` without modification.
    Otherwise:
    - `text_idx` can contain integers or feature names (strings), depending on input type.
    - `strategy` is normalized into `EmbedderParameters` via `get_embedder_parameters`.
    - A corresponding embedder function is selected from `EMBEDDING_METHOD_MAPPING`.
    - The selected text columns are replaced with zeros in the returned `features`.

    Args:
        features (ArrayType): Feature matrix `(n_samples, n_features)`.
        text_idx (IndexType): Indices or names of text columns to embed.
        strategy (Union[EmbedderParameters, Dict]): Embedding configuration.
        features_names (Optional[List[str]]): Required when `text_idx` contains names.

    Returns:
        Tuple[Any, Any, ArrayType]: `(embeddings, text_idx_resolved, features_updated)`.
            `embeddings` is the result returned by the selected embedding function
            (often a torch tensor), and `features_updated` is a copy of `features`
            with text columns set to zeros.
    """
    xp = backend.xp
    device = backend.device

    if text_idx is None:
        return None, None, features

    strategy = get_embedder_parameters(strategy)

    text_idx = get_idx_from_features_names(text_idx, features_names)

    text_features = features[:, text_idx]

    try:
        embedder_fn = EMBEDDING_METHOD_MAPPING[strategy.method]
    except KeyError:
        raise ValueError(f"Unknown embedding method: {strategy.method}")

    embeddings = embedder_fn(text_features, strategy)

    if strategy.device.type != device.type:
        embeddings = embeddings.to(device)

    features[:, text_idx] = xp.zeros(
        (features.shape[0], len(text_idx)),
        dtype=float
    )

    return embeddings, text_idx, features


def encode_target(target: ArrayType) -> Tuple[ArrayType, Any]:
    """
    Encode categorical target values into numeric representation.

    The function uses an encoder from `ENCODER_MAPPING` with
    `EncodingStrategyEnum.label` strategy.

    Args:
        target (ArrayType): Target array.

    Returns:
        Tuple[ArrayType, Any]:
            - encoded_target (ArrayType): Encoded target values.
            - target_encoder (Any): Fitted encoder instance, or `None` if `target` is empty.
    """

    if target is None or target.shape[0] == 0:
        return target, None

    encoder = ENCODER_MAPPING[EncodingStrategyEnum.label]()
    target = encoder.fit_transform(target, [0])

    return target, encoder


def force_categorical_determination(table: ArrayType) -> IndexType:
    """
    Detect categorical feature columns by checking for string/object dtypes.

    Args:
        table (ArrayType): Feature matrix `(n_samples, n_features)`.

    Returns:
        IndexType: Indices of detected categorical columns, or `None` if no categorical
            columns are found.
    """
    pd_backend = backend.pd

    categorical_ids = []

    for column_id, column in enumerate(table.T):
        series = pd_backend.Series(column)
        if str(series.dtype) in ("object", "string"):
            categorical_ids.append(column_id)

    if len(categorical_ids) == 0:
        return None

    categorical_ids = convert_idx_to_array(categorical_ids)
    return categorical_ids


def process_user_stratedy_encoding(strategy: EncodingStrategyType,
                                   features_names: List[str]) -> List[CategoricalEncodingDecision]:
    """
    Convert user-provided categorical encoding strategy into a list of decisions.

    Args:
        strategy (EncodingStrategyType): Either a single `CategoricalEncodingDecision`
            or a dict mapping encoding strategy name to indices/names of categorical columns.
        features_names (List[str]): Feature names used when the strategy specifies columns by name.

    Returns:
        List[CategoricalEncodingDecision]: A list of resolved encoding decisions.
    """

    if isinstance(strategy, CategoricalEncodingDecision):
        return [strategy]

    strategy_list = []

    for strategy_name, idx in strategy.items():
        idx = convert_idx_to_array(idx)
        idx = get_idx_from_features_names(idx, features_names)
        strategy_list.append(
            CategoricalEncodingDecision(idx, EncodingStrategyEnum(strategy_name))
        )

    return strategy_list


def choose_categorical_encoding(
    data: ArrayType,
    categorical_idx: IndexType = None,
    user_strategy: EncodingStrategyType = None,
    features_names: Optional[List[str]] = None,
    state: StateEnum = StateEnum.FIT
) -> tuple[Optional[List[CategoricalEncodingDecision]], IndexType]:
    """
    Determine categorical feature columns and produce encoding decisions.

    In `FIT` mode the function either:
    - Uses `user_strategy` if it is provided (dict or `CategoricalEncodingDecision`), or
    - Infers `categorical_idx` automatically from dtypes (string/object columns),
      or converts it from provided indices/names.

    In non-`FIT` mode it expects `user_strategy` to already be a list of decisions.

    Args:
        data (ArrayType): Feature matrix `(n_samples, n_features)`.
        categorical_idx (IndexType): Categorical column indices or names (optional).
        user_strategy (EncodingStrategyType): User strategy (optional).
        features_names (Optional[List[str]]): Feature names for name-to-index conversion.
        state (StateEnum): Whether we are in fitting or transformation stage.

    Returns:
        tuple[Optional[List[CategoricalEncodingDecision]], IndexType]:
            - decisions: Resolved list of encoding decisions, or `None` if no categorical columns exist.
            - non_categorical_idx: Indices of columns that are not categorical.
    """

    xp = backend.xp

    if state == StateEnum.FIT:

        if isinstance(user_strategy, Dict) or isinstance(user_strategy,
                                                         CategoricalEncodingDecision):
            decisions = process_user_stratedy_encoding(user_strategy, features_names)
            categorical_idx = xp.array(
                [col for dec in decisions for col in dec.categorical_columns]
            )
            non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)
            return decisions, non_categorical_idx

        elif user_strategy is not None:
            raise ValueError(
                f"User encoding strategy must be Dict or CategoricalEncodingDecision, got {type(user_strategy)}")

        if categorical_idx is not None:
            categorical_idx = get_idx_from_features_names(categorical_idx, features_names)
        else:
            categorical_idx = force_categorical_determination(data)

        if categorical_idx is None:
            return None, xp.arange(data.shape[1])

        non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)
        strategy = (
            EncodingStrategyEnum(user_strategy)
            if user_strategy is not None
            else EncodingStrategyEnum.label
        )

        decisions = [CategoricalEncodingDecision(categorical_idx, strategy)]

    else:
        decisions = user_strategy
        if decisions is None:
            return None, xp.arange(data.shape[1])
        categorical_idx = xp.array(
            [col for dec in decisions for col in dec.categorical_columns]
        )

    non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)

    return decisions, non_categorical_idx


def apply_categorical_encoding(data: ArrayType,
                               decision: CategoricalEncodingDecision) -> Tuple[ArrayType, CategoricalEncodingDecision]:
    """
    Apply one categorical encoding decision to the provided dataset.

    If the decision already has an `encoder`, it will use it to `transform`.
    Otherwise, it creates the encoder from `ENCODER_MAPPING` and performs `fit_transform`.

    Args:
        data (ArrayType): Feature matrix.
        decision (CategoricalEncodingDecision): Encoding decision containing categorical column indices and strategy.

    Returns:
        Tuple[ArrayType, CategoricalEncodingDecision]: `(encoded_features, updated_decision)`.
    """

    if decision.categorical_columns is None:
        return data, decision

    if decision.encoder is not None:
        features = decision.encoder.transform(data)
        return features, decision

    try:
        decision.encoder = ENCODER_MAPPING[decision.strategy]()
    except KeyError:
        raise ValueError(f"Unknown encoding strategy: {decision.strategy}")

    features = decision.encoder.fit_transform(data,
                                              decision.categorical_columns)

    return features, decision


def encode_categorical_features(data: ArrayType,
                                decisions: Optional[List[CategoricalEncodingDecision]],
                                non_categorical_idx: IndexType
                                ) -> Tuple[ArrayType, Optional[List[CategoricalEncodingDecision]]]:
    """
    Encode all categorical columns and concatenate them with non-categorical features.

    Args:
        data (ArrayType): Feature matrix `(n_samples, n_features)`.
        decisions (Optional[List[CategoricalEncodingDecision]]): List of categorical encoding decisions.
        non_categorical_idx (IndexType): Indices of non-categorical feature columns.

    Returns:
        Tuple[ArrayType, Optional[List[CategoricalEncodingDecision]]]:
            - result_data: Concatenated features `(n_samples, n_new_features)`.
            - new_decisions: Updated decisions with fitted encoders, or `None` if `decisions` is `None`.
    """
    xp = backend.xp

    if decisions is None:
        return data, None

    result_data = data[:, non_categorical_idx].copy()
    new_decisions = []

    for dec in decisions:
        encoded_features, dec_new = apply_categorical_encoding(data, dec)
        result_data = xp.hstack((result_data, encoded_features))
        new_decisions.append(dec_new)

    return result_data, new_decisions


def encode_torch_tensors(features: torch.Tensor,
                         user_strategy: EncodingStrategyType,
                         categorical_idx: IndexType,
                         state: StateEnum = StateEnum.FIT,
                         features_names: Optional[List[str]] = None
                         ) -> Tuple[torch.Tensor, Optional[List[CategoricalEncodingDecision]]]:
    """
    Perform categorical encoding and return encoded torch tensor features.

    This function converts the input `features` tensor to backend array format (NumPy/CuPy),
    selects categorical columns/decisions with `choose_categorical_encoding`, applies encoders,
    and converts the result back into a `torch.Tensor`.

    Args:
        features (torch.Tensor): Input feature tensor.
        user_strategy (EncodingStrategyType): Categorical encoding strategy (optional).
        categorical_idx (IndexType): Categorical column indices or names (optional).
        state (StateEnum): Whether we are in FIT or transform stage.
        features_names (Optional[List[str]]): Feature names for name-to-index conversion.

    Returns:
        Tuple[torch.Tensor, Optional[List[CategoricalEncodingDecision]]]:
            - encoded_features (torch.Tensor): Encoded float32 tensor.
            - decisions: Updated encoding decisions (with fitted encoders) or `None`.
    """

    if (user_strategy is None) and (categorical_idx is None):
        return features, None

    xp = backend.xp

    categorical_idx = convert_idx_to_array(categorical_idx)
    features_names = convert_to_list(features_names)

    features = xp.asarray(features)

    decisions, non_categorical_idx = choose_categorical_encoding(
        features, categorical_idx, user_strategy, features_names, state
    )

    features, decisions = encode_categorical_features(features, decisions, non_categorical_idx)

    features = to_tensor(features, dtype=torch.float32)

    return features, decisions


def get_target_and_features(
    features: ArrayType,
    target: ArrayType,
    features_names: Optional[List[str]] = None,
    target_idx: IndexType = None,
    state: StateEnum = StateEnum.FIT,
    data_type: DataTypesEnum = DataTypesEnum.tabular
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

    if data_type == DataTypesEnum.ts:
        return features, target, None

    xp = backend.xp

    if state == StateEnum.FIT:
        if target is not None:
            target = xp.array(target)
        else:
            if target_idx is not None:
                target_idx = get_idx_from_features_names(target_idx, features_names)
                target = features[:, target_idx].copy()
            else:
                target = features[:, -1].copy()
                target_idx = xp.array([-1])

            features = xp.delete(features, target_idx, axis=1)

        # TODO: replece encode target
        target = atleast_n_dimensions(target, 2)
        target = replace_missing_with_nan(target)
        target, target_encoder = encode_target(target)

        features, target = _drop_rows_with_nan_in_target(features, target)

        return features, target, target_encoder

    return features, None, None


def transform_to_tensor(features: ArrayType,
                        target: ArrayType,
                        text_tensors: Optional[torch.Tensor] = None,
                        text_idx: Optional[IndexType] = None,
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
    xp = backend.xp

    if text_idx is not None:
        features = xp.delete(features, text_idx, axis=1)

    if features.shape[1] != 0:
        features = to_tensor(features, dtype=torch.float32)
        if text_tensors is not None:
            features = torch.cat((features, text_tensors), dim=1)
    else:
        features = text_tensors

    if ts_init_shape is not None:
        features = features.reshape(ts_init_shape)

    target = to_tensor(target, dtype=torch.float32)

    return features, target


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
    xp = backend.xp
    device = backend.device

    if array is None:
        return None

    array = array.astype(xp.float64)
    return torch.tensor(array, dtype=dtype, device=device)
