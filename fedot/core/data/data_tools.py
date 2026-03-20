import os
import numpy as np
import numbers
import torch
import logging
from typing import Optional, Union, List, Tuple, Dict, Iterable, Any

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
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False


def get_device_from_str(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def convert_bytes(x: np.ndarray) -> np.ndarray:
    # Conversion of target values to float or str
    try:
        x = np.char.decode(x, encoding='utf-8')
    except:
        pass
    try:
        x = x.astype('float')
    except ValueError:
        x = x.astype(str)
    return x


def get_values_from_df(df: PandasType) -> ArrayType:
    try:
        features = df.values
        return features
    except Exception as e:
        features = df.to_pandas().values
        logger.info(f"Using pandas instead of cudf. Failed to get values from cudf DataFrame.")
        return features


def replace_missing_with_nan(arr: ArrayType) -> ArrayType:
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
    Drop rows where target contains NaN in any target column.
    Works for both numpy and cupy backends.
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
    Returns a view of the ``data`` with at least ``ndim`` dimensions

    :param data: ndarray which dimensional size should be set to at least ``ndim``
    :param ndim: number of required axes to have in ``data``

    :return: ``data`` expanded from the last axis to the provided ``ndim`` size if it doesn't satisfy it
    """
    xp = backend.xp

    while data.ndim < ndim:
        data = xp.expand_dims(data, axis=-1)
    return data



def convert_idx_to_array(idx: IndexType) -> IndexType:
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
    if isinstance(idx, list) or idx is None:
        return idx
    elif isinstance(idx, np.ndarray) or isinstance(idx, backend.xp.ndarray):
        return idx.tolist()
    else:
        return [idx]



def get_idx_from_features_names(idx: IndexType, 
                                features_names: Optional[List[str]]) -> IndexType:
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
    Encode categorical target values and ensure numeric dtype.
    """

    if target is None or target.shape[0] == 0:
        return target, None
    
    encoder = ENCODER_MAPPING[EncodingStrategyEnum.label]()
    target = encoder.fit_transform(target, [0])

    return target, encoder


def force_categorical_determination(table: ArrayType) -> IndexType:
    """Find string columns using a unified approach for CPU/GPU backends."""
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
            raise ValueError(f"User encoding strategy must be Dict or CategoricalEncodingDecision, got {type(user_strategy)}")

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
        categorical_idx = xp.array(
            [col for dec in decisions for col in dec.categorical_columns]
        )

    non_categorical_idx = xp.setdiff1d(xp.arange(data.shape[1]), categorical_idx)
    
    return decisions, non_categorical_idx


def apply_categorical_encoding(data: ArrayType, 
                               decision: CategoricalEncodingDecision) -> Tuple[ArrayType, CategoricalEncodingDecision]:

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
    """Function for getting target and features from numpy array"""

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
    xp = backend.xp
    device = backend.device

    if array is None:
        return None

    array = array.astype(xp.float64)
    return torch.tensor(array, dtype=dtype, device=device)
