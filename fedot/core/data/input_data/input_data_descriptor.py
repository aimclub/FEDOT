from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from fedot.core.data.common.compatibility_rules import build_data_type_compatibility
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task


@dataclass(frozen=True)
class InputDataDescriptor:
    task: Task
    original_data_type: DataTypesEnum
    tensor_canonical_data_type: DataTypesEnum
    input_compatible_data_type: DataTypesEnum
    idx: np.ndarray
    features_names: Optional[np.ndarray]
    categorical_idx: Optional[np.ndarray]
    target_is_present: bool


def normalize_optional_numpy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.copy()
    return np.asarray(value)


def build_input_data_descriptor(input_data) -> InputDataDescriptor:
    compatibility = build_data_type_compatibility(input_data.data_type)

    return InputDataDescriptor(
        task=input_data.task,
        original_data_type=input_data.data_type,
        tensor_canonical_data_type=compatibility.tensor_canonical,
        input_compatible_data_type=compatibility.input_compatible,
        idx=np.asarray(input_data.idx).copy(),
        features_names=normalize_optional_numpy_array(input_data.features_names),
        categorical_idx=normalize_optional_numpy_array(input_data.categorical_idx),
        target_is_present=input_data.target is not None,
    )
