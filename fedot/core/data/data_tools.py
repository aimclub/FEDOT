import os
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any

from fedot.core.backend.backend import Backend
from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass, field

import torch
from torch import Tensor

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


def get_target_and_features(
    features,
    target,
    features_names,
    target_idx: Optional[Union[int, np.ndarray]],
    state: Optional[str],
):
    """Function for getting target and features from numpy array"""
    xp = Backend.xp

    if state == "fit":
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
        target = encode_target(target)
        target = replace_missing_with_nan(target)

        features, target = _drop_rows_with_nan_in_target(features, target)

        return features, target

    return features, None




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


# TODO: replece it
def encode_target(target):
    """
    Encode categorical target values and ensure numeric dtype.
    """
    xp = Backend.xp

    if target is None or target.shape[0] == 0:
        return target

    target = xp.asarray(target)

    if target.dtype.kind in {"U", "S"}:
        target_flat = target.flatten()
        _, codes = xp.unique(target_flat, return_inverse=True)
        codes = codes.astype(xp.int64)
        return codes.reshape(-1, 1)

    if target.dtype == object:
        if isinstance(target.flat[0], str):
            target_flat = target.flatten()
            _, codes = xp.unique(target_flat, return_inverse=True)
            return codes.astype(xp.int64).reshape(-1, 1)

        try:
            return target.astype(xp.int64)
        except Exception:
            return target.astype(xp.float32)

    if target.dtype.kind in {"i", "u"}:
        return target.astype(xp.int64)

    if target.dtype.kind == "f":
        return target.astype(xp.float32)

    return target
