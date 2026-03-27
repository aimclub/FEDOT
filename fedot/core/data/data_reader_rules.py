from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from fedot.core.data.data_tools import convert_bytes


@dataclass(frozen=True)
class ArffTargetResolution:
    target_idx: Optional[int]


def infer_arff_target_idx(data_array: np.ndarray) -> Optional[int]:
    if isinstance(convert_bytes(data_array[-1])[0], str):
        return -1
    if isinstance(convert_bytes(data_array[0])[0], str):
        return 0
    return None


def resolve_arff_target_idx(
    target_idx: Optional[Union[int, str]],
    field_names: Sequence[str],
    data_array: np.ndarray,
) -> ArffTargetResolution:
    if isinstance(target_idx, str):
        if target_idx not in field_names:
            raise ValueError(f"Unknown ARFF target column: {target_idx}")
        return ArffTargetResolution(target_idx=field_names.index(target_idx))

    if isinstance(target_idx, int):
        if not (-len(field_names) <= target_idx < len(field_names)):
            raise ValueError(f"ARFF target_idx out of range: {target_idx}")
        return ArffTargetResolution(target_idx=target_idx)

    if target_idx is None:
        return ArffTargetResolution(target_idx=infer_arff_target_idx(data_array))

    raise TypeError(f"target_idx must be int, str, or None, got {type(target_idx)}")


def split_arff_features_and_target(
    data_array: np.ndarray,
    target_idx: Optional[int],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if target_idx is None:
        return convert_bytes(data_array), None

    target = convert_bytes(data_array[target_idx])
    features = convert_bytes(np.delete(data_array, target_idx, axis=0))
    return features, target
