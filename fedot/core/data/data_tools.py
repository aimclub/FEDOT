import os
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any

from fedot.core.backend.backend import Backend

from fedot.core.data.tools import StateEnum
from fedot.preprocessing.categorical_encoding import (LabelEncoder, 
    OneHotEncoder, EncodingStrategyEnum, CategoricalEncodingDecision)

import torch

import logging

logger = logging.getLogger(__name__)


def is_existed_csv_path(path: str) -> bool:
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return True
    return False


def get_device_from_str(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def convert_bytes(x):
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


def get_values_from_df(df):
    try:
        features = df.values
        return features
    except Exception:
        raise ValueError("Fedot preprocessing doesn't support categorical data in gpu mode")


def replace_missing_with_nan(arr):
    xp = Backend.xp
    pd_backend = Backend.pd
    backend_name = Backend.name

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

    def _normalize(x):
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


def _drop_rows_with_nan_in_target(features, target):
    """
    Drop rows where target contains NaN in any target column.
    Works for both numpy and cupy backends.
    """
    xp = Backend.xp

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


def atleast_n_dimensions(data, ndim):
    """
    Returns a view of the ``data`` with at least ``ndim`` dimensions

    :param data: ndarray which dimensional size should be set to at least ``ndim``
    :param ndim: number of required axes to have in ``data``

    :return: ``data`` expanded from the last axis to the provided ``ndim`` size if it doesn't satisfy it
    """
    xp = Backend.xp

    while data.ndim < ndim:
        data = xp.expand_dims(data, axis=-1)
    return data



def convert_idx_to_array(idx):
    xp = Backend.xp

    if isinstance(idx, xp.ndarray) or idx is None:
        return idx
    try:
        if isinstance(idx, int) or isinstance(idx, str):
            return xp.array([idx])
        else:
            return xp.array(idx)
    except Exception:
        return list(idx)


def get_idx_from_features_names(idx, features_names):
    xp = Backend.xp

    if isinstance(idx[0], xp.int_):
        return idx

    if features_names is None:
        raise ValueError(
            "Impossible to specify categorical features by name when the features_names are not specified"
        )

    try:
        if isinstance(idx[0], str):
            return xp.array([xp.where(features_names == name)[0][0] for name in idx])
    except Exception:
        raise ValueError(f"Failed to get index from features names: {idx}")
    

def encode_target(target):
    """
    Encode categorical target values and ensure numeric dtype.
    """

    if target is None or target.shape[0] == 0:
        return target, None
    
    encoder = LabelEncoder()
    target = encoder.fit_transform(target, [0])

    return target, encoder


def force_categorical_determination(table):
    """Find string columns using a unified approach for CPU/GPU backends."""
    pd_backend = Backend.pd

    categorical_ids = []

    for column_id, column in enumerate(table.T):
        series = pd_backend.Series(column)
        if str(series.dtype) in ("object", "string"):
            categorical_ids.append(column_id)

    if len(categorical_ids) == 0:
        return None

    categorical_ids = convert_idx_to_array(categorical_ids)
    return categorical_ids


def process_user_stratedy_encoding(strategy: Union[Dict, CategoricalEncodingDecision], features_names):

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
    data,
    categorical_idx=None,
    user_strategy=None,
    features_names=None,
    state: StateEnum = StateEnum.FIT
) -> List[CategoricalEncodingDecision]:
    
    xp = Backend.xp
    
    if state == StateEnum.FIT:

        if isinstance(user_strategy, Dict) or isinstance(user_strategy, CategoricalEncodingDecision):
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


def apply_categorical_encoding(data, decision):

    if decision.categorical_columns is None:
        return data, decision
    
    if decision.encoder is not None:
        features = decision.encoder.transform(data)
        return features, decision

    if decision.strategy == EncodingStrategyEnum.label:
        decision.encoder = LabelEncoder()
        features = decision.encoder.fit_transform(data, decision.categorical_columns)

    elif decision.strategy == EncodingStrategyEnum.ohe:
        decision.encoder = OneHotEncoder()
        features = decision.encoder.fit_transform(data, decision.categorical_columns)

    return features, decision


def encode_categorical_features(data, decisions, non_categorical_idx):
    xp = Backend.xp

    if decisions is None:
        return data, None

    result_data = data[:, non_categorical_idx].copy()
    new_decisions = []

    for dec in decisions:
        encoded_features, dec_new = apply_categorical_encoding(data, dec)
        result_data = xp.hstack((result_data, encoded_features))
        new_decisions.append(dec_new)

    return result_data, new_decisions


def encode_torch_tensors(features, user_strategy, categorical_idx, state: StateEnum = StateEnum.FIT, features_names=None):

    if (user_strategy is None) and (categorical_idx is None):
        return features, None

    xp = Backend.xp

    categorical_idx = convert_idx_to_array(categorical_idx)
    features_names = convert_idx_to_array(features_names)

    features = xp.asarray(features)

    decisions, non_categorical_idx = choose_categorical_encoding(
        features, categorical_idx, user_strategy, features_names, state
    )

    features, decisions = encode_categorical_features(features, decisions, non_categorical_idx)

    features = to_tensor(features, dtype=torch.float32)

    return features, decisions


def get_target_and_features(
    features,
    target,
    features_names,
    target_idx: Optional[Union[int, np.ndarray]],
    state: StateEnum = StateEnum.FIT,
):
    """Function for getting target and features from numpy array"""
    xp = Backend.xp

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




def transform_to_tensor(features, target, text_tensors, text_idx, ts_init_shape):
    xp = Backend.xp

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


def to_tensor(array, dtype=None):
    xp = Backend.xp
    device = Backend.device

    if array is None:
        return None

    array = array.astype(xp.float64)
    return torch.tensor(array, dtype=dtype, device=device)
