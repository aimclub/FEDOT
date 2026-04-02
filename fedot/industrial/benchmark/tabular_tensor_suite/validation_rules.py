from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import BenchmarkStage


@dataclass(frozen=True)
class BenchmarkValidationResult:
    feature_rows: int | None
    idx_length: int | None
    target_length: int | None


class BenchmarkValidationError(ValueError):
    def __init__(self, stage: BenchmarkStage, message: str):
        super().__init__(message)
        self.stage = stage.value


def validate_row_accounting(data: Any, *, stage: BenchmarkStage, label: str) -> BenchmarkValidationResult:
    features = getattr(data, 'features', None)
    idx = getattr(data, 'idx', None)
    target = getattr(data, 'target', None)

    feature_rows = _resolve_feature_rows(features)
    idx_length = _resolve_length(idx)
    target_length = _resolve_length(target)

    if idx_length is None:
        raise BenchmarkValidationError(stage, f'{label}: idx is missing or has no length.')

    if feature_rows is not None and idx_length != feature_rows:
        raise BenchmarkValidationError(
            stage,
            f'{label}: idx length {idx_length} does not match feature rows {feature_rows}.',
        )

    if target is not None and target_length is not None and idx_length != target_length:
        raise BenchmarkValidationError(
            stage,
            f'{label}: idx length {idx_length} does not match target length {target_length}.',
        )

    return BenchmarkValidationResult(
        feature_rows=feature_rows,
        idx_length=idx_length,
        target_length=target_length,
    )


def validate_idx_round_trip(original_data: Any,
                            round_trip_data: Any,
                            *,
                            stage: BenchmarkStage,
                            label: str) -> None:
    original_idx = _to_numpy_copy(getattr(original_data, 'idx', None))
    round_trip_idx = _to_numpy_copy(getattr(round_trip_data, 'idx', None))

    if original_idx is None or round_trip_idx is None:
        raise BenchmarkValidationError(stage, f'{label}: idx is missing after tensor round-trip.')
    if not np.array_equal(original_idx, round_trip_idx):
        raise BenchmarkValidationError(
            stage,
            f'{label}: idx drift detected after tensor round-trip.',
        )


def _resolve_feature_rows(features: Any) -> int | None:
    if features is None:
        return None
    shape = getattr(features, 'shape', None)
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except TypeError:
            return None
    try:
        return int(len(features))
    except TypeError:
        return None


def _resolve_length(value: Any) -> int | None:
    if value is None:
        return None
    shape = getattr(value, 'shape', None)
    if shape is not None:
        try:
            if len(shape) >= 1:
                return int(shape[0])
        except TypeError:
            pass
    try:
        return int(len(value))
    except TypeError:
        return None


def _to_numpy_copy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value, copy=True)

