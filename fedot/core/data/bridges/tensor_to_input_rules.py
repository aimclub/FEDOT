from dataclasses import dataclass
from typing import Optional

import numpy as np

from fedot.core.data.common.compatibility_rules import to_input_compatible_data_type
from fedot.core.data.common.enums import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task


@dataclass(frozen=True)
class TensorDataInputBridgePlan:
    task: Task
    data_type: DataTypesEnum
    state: StateEnum
    idx: np.ndarray
    features: np.ndarray
    target: Optional[np.ndarray]
    features_names: Optional[np.ndarray]
    categorical_idx: Optional[np.ndarray]


def to_numpy_copy(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()

    return np.array(value, copy=True)


def normalize_tensor_bridge_state(state) -> StateEnum:
    if isinstance(state, StateEnum):
        return state
    return StateEnum(state)


def resolve_input_idx(idx, features: np.ndarray) -> np.ndarray:
    normalized_idx = to_numpy_copy(idx)
    if normalized_idx is not None:
        return normalized_idx
    return np.arange(len(features))


def normalize_input_bridge_target(target, state: StateEnum) -> Optional[np.ndarray]:
    if state is StateEnum.PREDICT:
        return None
    return to_numpy_copy(target)


def build_tensordata_input_bridge_plan(tensor_data) -> TensorDataInputBridgePlan:
    state = normalize_tensor_bridge_state(tensor_data.state)
    features = to_numpy_copy(tensor_data.features)
    if features is None:
        raise ValueError('TensorData features are required for conversion to InputData')

    return TensorDataInputBridgePlan(
        task=tensor_data.task,
        data_type=to_input_compatible_data_type(tensor_data.data_type),
        state=state,
        idx=resolve_input_idx(tensor_data.idx, features),
        features=features,
        target=normalize_input_bridge_target(tensor_data.target, state),
        features_names=to_numpy_copy(tensor_data.features_names),
        categorical_idx=to_numpy_copy(tensor_data.categorical_idx),
    )
