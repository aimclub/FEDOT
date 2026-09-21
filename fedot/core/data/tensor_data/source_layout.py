"""Normalize source axes, selectors and sample masks before fitting handlers."""
from copy import deepcopy
from numbers import Integral

import numpy as np
import torch

from fedot.core.backend.backend import Backend, torch_to_xp
from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
from fedot.core.data.tensor_data.contracts import (
    FeatureSchema, TensorDataContractError, as_list, column_indices,
    sample_index, select_rows, validate_shape, validate_target_rows,
)
from fedot.core.data.tensor_data.tools import replace_missing_with_np_nan, target_row_mask
from fedot.core.repository.dataset_types import DataTypesEnum


def owned_array(value, xp):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = torch_to_xp(value, xp)
    elif xp is np and hasattr(value, 'get'):
        value = value.get()
    return xp.array(value, copy=True)


def _strategy_indices(strategy, width, names, kept):
    if strategy is None:
        return None
    entries = [strategy] if isinstance(strategy, dict) else list(strategy)
    result = deepcopy(entries)
    for entry in result:
        if entry is None or entry.get('features_idx') is None:
            continue
        indices = column_indices(
            entry['features_idx'], width, names, 'features_idx')
        if any(pos not in kept for pos in indices):
            raise TensorDataContractError(
                'removed_column', 'features_idx', 'strategy selects a target column')
        entry['features_idx'] = [kept.index(pos) for pos in indices]
    return result


def _time_series(spec, xp):
    features, target = spec.features, spec.target
    orientation = spec.ts_orientation or TSOrientationEnum.wide
    if orientation == TSOrientationEnum.long:
        if features.ndim != 2 or features.shape[1] != 2:
            raise TensorDataContractError(
                'invalid_axes', 'features', 'long time series require term and value columns')
        terms = column_indices(spec.ts_terms_idx if spec.ts_terms_idx is not None else [0],
                               2, spec.features_names, 'ts_terms_idx')
        if len(terms) != 1:
            raise TensorDataContractError(
                'invalid_selector', 'ts_terms_idx', 'select exactly one term column')
        labels = xp.unique(features[:, terms[0]])
        series = [features[features[:, terms[0]] == label, 1 - terms[0]]
                  for label in labels]
        if len({len(values) for values in series}) != 1:
            raise TensorDataContractError(
                'invalid_axes', 'features', 'all long-format series must have equal length')
        features = xp.stack(series)
        if spec.idx is None:
            spec.idx = labels.get() if hasattr(labels, 'get') else labels
        spec.features_names = None
        spec.ts_terms_idx = terms
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    horizon = spec.ts_forecast_horizon
    if spec.target_idx is not None and (features.ndim != 2 or horizon is not None):
        raise TensorDataContractError('ambiguous_target', 'target_idx',
                                      'target columns require 2D series without horizon extraction')
    if horizon is not None:
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
            raise TensorDataContractError(
                'invalid_horizon', 'ts_forecast_horizon', 'expected a positive integer')
        if spec.state == StateEnum.FIT and not spec.without_target and target is None:
            if horizon >= features.shape[-1]:
                raise TensorDataContractError(
                    'invalid_horizon', 'ts_forecast_horizon', 'horizon must leave at least one context step')
            target = features[..., -horizon:].copy()
            features = features[..., :-horizon].copy()
    spec.ts_orientation = orientation
    spec.ts_init_shape = tuple(features.shape)
    return features, target


def prepare_source(spec):
    """Normalize a creator-owned spec; never retain views into caller arrays."""
    xp = Backend().xp
    is_ts = spec.data_type == DataTypesEnum.ts
    validate_shape(spec.features.shape, time_series=is_ts)
    features = spec.features
    if not is_ts and features.ndim == 1:
        features = features.reshape(-1, 1)
    spec.features = features
    if is_ts:
        features, target = _time_series(spec, xp)
    else:
        target = spec.target
    width = features.shape[1]
    names = as_list(
        spec.features_names) if spec.features_names is not None else None
    if names is not None and (len(names) != width or len(set(names)) != width):
        raise TensorDataContractError(
            'invalid_names', 'features_names', f'expected {width} unique feature names')
    idx = sample_index(spec.idx, len(features))
    categorical = column_indices(
        spec.categorical_idx, width, names, 'categorical_idx')
    numerical = column_indices(
        spec.numerical_idx, width, names, 'numerical_idx')
    if set(categorical) & set(numerical):
        raise TensorDataContractError(
            'overlapping_indices', 'categorical_idx', 'categorical and numerical columns overlap')
    extracted = []
    if target is None and not spec.without_target:
        if spec.target_idx is not None:
            extracted = column_indices(spec.target_idx, width, names)
        elif spec.state == StateEnum.FIT and not is_ts:
            extracted = [width - 1]
        if extracted:
            target = features[:, extracted].copy()
            features = xp.delete(features, extracted, axis=1)
    kept = [pos for pos in range(width) if pos not in extracted]
    if not kept:
        raise TensorDataContractError(
            'invalid_axes', 'features', 'target extraction leaves no feature columns')
    if set(extracted) & (set(categorical) | set(numerical)):
        raise TensorDataContractError(
            'removed_column', 'categorical_idx', 'feature selectors include a target column')
    features = replace_missing_with_np_nan(features)
    if target is not None:
        if target.ndim == 1:
            target = target.reshape(
                1, -1) if is_ts and len(features) == 1 else target.reshape(-1, 1)
        validate_target_rows(target, len(features))
        target = replace_missing_with_np_nan(target)
        mask = target_row_mask(target)
        if spec.state == StateEnum.FIT:
            features, target = features[mask], target[mask]
            idx = select_rows(idx, mask)
        elif not bool(mask.all()):
            raise TensorDataContractError(
                'missing_target', 'target', 'prediction targets must not remove samples')
    names = [names[pos] for pos in kept] if names is not None else None
    state = spec.preparation_state
    schema = FeatureSchema(tuple(features.shape[1:]), None if names is None else tuple(names),
                           tuple(enumerate(kept)), tuple(categorical),
                           variable_time=is_ts and spec.ts_forecast_horizon is not None)
    if state is not None:
        state.schema.validate(features.shape, names)
        if not (is_ts and features.ndim == 2 and state.schema.variable_time):
            schema = state.schema
    else:
        for option in ('encoding_strategy', 'embedding_strategy', 'custom_strategy'):
            setattr(spec, option, _strategy_indices(getattr(spec, option), width,
                                                    spec.features_names, kept))
    spec.features, spec.target, spec.idx = features, target, idx
    spec.features_names = names
    spec.idx_mapping = dict(schema.mapping)
    if is_ts:
        spec.ts_init_shape = tuple(features.shape)
    return schema
