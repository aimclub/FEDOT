"""Local, side-effect-free rules for the TensorData creation boundary."""
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Optional

import numpy as np
import torch

from fedot.core.backend.array_conversions import as_numpy, as_torch


class TensorDataContractError(ValueError):
    """Invalid data with a stable category and the offending public field."""

    def __init__(self, code: str, field: str, message: str):
        self.code = code
        self.field = field
        super().__init__(f'{field}: {message}')


@dataclass(frozen=True)
class CategoricalData:
    """Original categories beside current numerical tensors, without a device transfer.

    Model adapters choose how to combine these blocks for their library.
    Missing category values are normalized to np.nan by source preparation.
    """

    numerical_features: torch.Tensor
    categorical_features: np.ndarray
    features_names: Optional[list] = None


@dataclass(frozen=True)
class FeatureSchema:
    """Feature layout after target extraction, before fitted transformations."""

    shape: tuple[int, ...]
    names: Optional[tuple[Any, ...]]
    mapping: tuple[tuple[int, int], ...]
    categorical: tuple[int, ...]
    variable_time: bool = False

    def validate(self, shape, names):
        trailing = tuple(shape[1:])
        expected = self.shape
        if self.variable_time:
            trailing, expected = trailing[:-1], expected[:-1]
        if len(shape) != len(self.shape) + 1 or trailing != expected:
            raise TensorDataContractError(
                'schema_mismatch', 'features',
                f'expected trailing shape {self.shape}, got {tuple(shape[1:])}')
        if self.names is not None and tuple(names or ()) != self.names:
            raise TensorDataContractError(
                'schema_mismatch', 'features_names',
                'prediction columns must match training columns in the same order')


@dataclass(frozen=True)
class FittedPreparationStep:
    """Owned fitted handler snapshot, not a plan to fit a new handler."""

    step: str
    indices: tuple[int, ...]
    handler_state: bytes = field(repr=False)
    source_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreparationState:
    """Training-owned preparation; prediction works on private handler copies.

    No source arrays are retained. ``backend_name`` is the backend that fitted
    handlers use, which can differ from the destination tensor device.
    """

    schema: FeatureSchema
    steps: tuple[FittedPreparationStep, ...]
    plan_state: bytes = field(repr=False)
    plan_hash: str
    backend_name: str
    input_hash: str
    # A merger can rename branch-local categorical sources without changing
    # original/target preparation. None means the original encoding steps.
    categorical_steps: Optional[tuple[FittedPreparationStep, ...]] = None

    @property
    def plan(self):
        """Return a private plan restored from the trusted fitted snapshot."""
        import cloudpickle
        return cloudpickle.loads(self.plan_state)

    @property
    def encoding_steps(self):
        if self.categorical_steps is not None:
            return self.categorical_steps
        return tuple(step for step in self.steps if step.step == 'encoding')


def as_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (str, Integral)):
        return [value]
    return list(value)


def column_indices(value, width: int, names=None, field='target_idx') -> list[int]:
    """Resolve ordered selectors; negative positions are canonicalized once.

    Boolean masks are not column positions. Mixed names/positions, duplicates
    (including negative aliases), unknown names and out-of-range indices fail.
    """
    values = as_list(value)
    if not isinstance(values, list):
        values = [values]
    if not values:
        return []
    integers = all(isinstance(x, Integral) and not isinstance(
        x, (bool, np.bool_)) for x in values)
    strings = all(isinstance(x, str) for x in values)
    if not integers and not strings:
        raise TensorDataContractError(
            'invalid_selector', field, 'use only integer positions or only names')
    if strings:
        names = as_list(names)
        if len(names) != width or len(set(names)) != width:
            raise TensorDataContractError(
                'invalid_names', 'features_names', 'unique column names are required')
        unknown = [x for x in values if x not in names]
        if unknown:
            raise TensorDataContractError(
                'unknown_column', field, f'unknown columns: {unknown}')
        resolved = [names.index(x) for x in values]
    else:
        if any(x < -width or x >= width for x in values):
            raise TensorDataContractError(
                'index_bounds', field, f'positions must be in [-{width}, {width})')
        resolved = [int(x) % width for x in values]
    if len(set(resolved)) != len(resolved):
        raise TensorDataContractError(
            'duplicate_selector', field, 'column positions must be unique')
    return resolved


def validate_shape(shape, *, time_series=False):
    allowed = (1, 2, 3) if time_series else (1, 2)
    if len(shape) not in allowed or any(size == 0 for size in shape):
        layout = '(time,), (samples, time), or (samples, channels, time)' if time_series else '(samples,) or (samples, features)'
        raise TensorDataContractError(
            'invalid_axes', 'features', f'expected nonempty {layout}; got {tuple(shape)}')


def validate_sample_index(idx, count: int):
    """Inspect row-label shape without allocating a tensor copy."""
    ndim = idx.ndim if hasattr(idx, 'ndim') else np.ndim(idx)
    if ndim != 1 or len(idx) != count:
        raise TensorDataContractError(
            'row_alignment', 'idx', f'expected {count} one-dimensional row labels')


def sample_index(idx, count: int, device=None):
    """Own row labels without interpreting numeric labels as positions."""
    if idx is None:
        return torch.arange(count, dtype=torch.int64, device=device)
    if isinstance(idx, torch.Tensor):
        result = idx.detach().to(device=device, copy=True)
    else:
        result = np.array(idx, copy=True)
    validate_sample_index(result, count)
    return result


def select_rows(value, mask):
    """Apply a strict boolean row mask, never flattening a higher-rank mask."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        mask = as_torch(mask, device=value.device)
        if mask.dtype != torch.bool or mask.ndim != 1 or len(mask) != len(value):
            raise TensorDataContractError(
                'invalid_mask', 'mask', 'expected one boolean per sample')
        return value[mask]
    mask = as_numpy(mask)
    if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(value):
        raise TensorDataContractError(
            'invalid_mask', 'mask', 'expected one boolean per sample')
    return value[mask]


def validate_target_rows(target, count: int):
    if target is not None and (target.ndim == 0 or target.shape[0] != count):
        raise TensorDataContractError(
            'row_alignment', 'target', f'expected {count} samples')


def validate_runtime_data(data):
    """Validate prepared arrays without changing data or selecting a backend."""
    shape = getattr(data.features, 'shape', ())
    if len(shape) not in (1, 2, 3) or any(size == 0 for size in shape):
        raise TensorDataContractError(
            'invalid_axes', 'features', f'invalid prepared shape {tuple(shape)}')
    if data.idx is not None:
        validate_sample_index(data.idx, shape[0])
    validate_target_rows(data.target, shape[0])
    if data.categorical_features is None:
        if data.categorical_source_idx:
            raise TensorDataContractError('missing_categories', 'categorical_features', 'source IDs require values')
    elif (data.categorical_features.ndim != 2
          or data.categorical_features.shape != (shape[0], len(data.categorical_source_idx))):
        raise TensorDataContractError(
            'row_alignment', 'categorical_features', 'expected one row per sample and one column per source ID')
    if len(set(data.categorical_source_idx)) != len(data.categorical_source_idx):
        raise TensorDataContractError('duplicate_selector', 'categorical_source_idx', 'source IDs must be unique')
    if data.predict is not None and (data.predict.ndim == 0 or data.predict.shape[0] != shape[0]):
        raise TensorDataContractError(
            'row_alignment', 'predict', 'predictions must align with samples')
    if isinstance(data.features, torch.Tensor):
        for name in ('target', 'predict', 'idx'):
            tensor = getattr(data, name)
            if isinstance(tensor, torch.Tensor) and tensor.device != data.features.device:
                raise TensorDataContractError(
                    'device_mismatch', name, 'tensor fields must share one device')
