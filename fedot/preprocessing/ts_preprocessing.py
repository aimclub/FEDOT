from dataclasses import dataclass
from fedot.core.backend.backend import Backend
from typing import Optional

from fedot.core.data.data_tools import get_idx_from_features_names
from fedot.core.data.tensordata import IndexType

from fedot.core.data.tools import TSOrientationEnum, StateEnum

def long_to_wide(features, features_names=None, terms_idx=None):
    xp = Backend.xp

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


def check_multichannel_ts(features):
    xp = Backend.xp

    if features.ndim == 1:
        features = xp.expand_dims(features, axis=0)
        init_shape = None
    elif features.ndim == 2:
        B, T = features.shape
        init_shape = None
    elif features.ndim == 3:
        B, C, T = features.shape
        features = features.reshape(B * C, T)
        init_shape = (B, C, T)
    elif features.ndim > 3:
        raise ValueError("Multichannel time series must not have more than 3 dimensions")

    return features, init_shape


def process_ts_data(
    features,
    target=None,
    features_names=None,
    state: StateEnum = StateEnum.FIT,
    ts_orientation: Optional[TSOrientationEnum] = None,
    terms_idx: int = None,
    forecast_horizon: int = None,
):
    features, init_shape = check_multichannel_ts(features)

    if ts_orientation is None:
        ts_orientation = TSOrientationEnum.wide
    elif ts_orientation == TSOrientationEnum.long:
        features, terms_idx = long_to_wide(features, features_names, terms_idx)

    if state == StateEnum.FIT and forecast_horizon is not None:
        target = features[features.shape[1] - forecast_horizon :, :]
        features = features[:-forecast_horizon, :]

    return features, target, init_shape, terms_idx
