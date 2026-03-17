from dataclasses import dataclass
from fedot.core.backend.backend import Backend
from typing import Optional

from fedot.core.data.data_tools import get_idx_from_features_names
from fedot.core.data.tensordata import IndexType

def split_long_array(features, features_names, terms_idx: IndexType = None):
    xp = Backend.xp

    if terms_idx is None:
        terms_idx = xp.array(range(features.shape[1] - 1))
    terms_idx = get_idx_from_features_names(terms_idx, features_names)[0]

    unique_labels = xp.unique(features[:, terms_idx])
    split_arrays = xp.array(
        [features[features[:, terms_idx] == label, :terms_idx] for label in unique_labels]
    )
    return split_arrays, terms_idx


def check_multichannel_ts(features):
    xp = Backend.xp

    if features.ndim == 1:
        features = xp.expand_dims(features, axis=0)
        init_shape = (1, features.shape[1])
    elif features.ndim == 2:
        B, T = features.shape
        init_shape = (B, T)
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
    state="fit",
    ts_orientation: Optional[str] = None,
    terms_idx: int = None,
    forecast_horizon: int = None,
):
    features, init_shape = check_multichannel_ts(features)

    if ts_orientation is None:
        ts_orientation = "wide"
    elif ts_orientation == "long":
        features, terms_idx = split_long_array(features, features_names, terms_idx)

    if state == "fit" and forecast_horizon is not None:
        target = features[features.shape[1] - forecast_horizon :, :]
        features = features[:-forecast_horizon, :]

    return features, target, init_shape, terms_idx
