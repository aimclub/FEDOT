from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from fedot.core.data.input_data.input_data_descriptor import InputDataDescriptor, normalize_optional_numpy_array
from fedot.core.data.common.enums import StateEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task


@dataclass(frozen=True)
class InputDataTensorBridgePlan:
    task: Task
    data_type: DataTypesEnum
    state: StateEnum
    target: Optional[np.ndarray]
    features_names: Optional[np.ndarray]
    categorical_idx: Optional[np.ndarray]


def normalize_bridge_state(state: Union[StateEnum, str]) -> StateEnum:
    if isinstance(state, StateEnum):
        return state
    if isinstance(state, str):
        return StateEnum(state)
    raise TypeError(f'state must be StateEnum or str, got {type(state)}')


def normalize_bridge_target(target: Any, state: StateEnum) -> Optional[np.ndarray]:
    if state == StateEnum.PREDICT or target is None:
        return None
    if isinstance(target, np.ndarray):
        return target.copy()
    return np.asarray(target)


def build_input_data_tensor_bridge_plan(
    descriptor: InputDataDescriptor,
    target: Any,
    state: Union[StateEnum, str] = StateEnum.FIT,
) -> InputDataTensorBridgePlan:
    normalized_state = normalize_bridge_state(state)
    return InputDataTensorBridgePlan(
        task=descriptor.task,
        data_type=descriptor.tensor_canonical_data_type,
        state=normalized_state,
        target=normalize_bridge_target(target, normalized_state),
        features_names=normalize_optional_numpy_array(
            descriptor.features_names),
        categorical_idx=normalize_optional_numpy_array(
            descriptor.categorical_idx),
    )
