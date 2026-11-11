"""Pure implementation selection; discovery, imports and fitting belong to the caller."""
from dataclasses import dataclass
from typing import FrozenSet, Optional, TYPE_CHECKING

import torch

from fedot.core.repository.operation_implementation import DeviceType, ExecutionPlatform, StrategyReference
from fedot.core.repository.tasks import TaskTypesEnum, compatible_task_types

if TYPE_CHECKING:
    from fedot.core.repository.operation_types_repository import OperationMetaInfo


@dataclass(frozen=True)
class ExecutionEnvironment:
    platform: ExecutionPlatform
    available_devices: FrozenSet[DeviceType]
    available_dependencies: FrozenSet[str]


@dataclass(frozen=True)
class OperationExecutionResolution:
    implementation_id: str
    device: torch.device
    strategy: StrategyReference

    @property
    def cache_key(self) -> str:
        return f'{self.implementation_id}:{self.device}:{self.strategy.module}.{self.strategy.name}'


def find_operation_execution(operation: 'OperationMetaInfo', task_type: TaskTypesEnum,
                             data_device: torch.device,
                             environment: ExecutionEnvironment,
                             multi_output: bool = False) -> Optional[OperationExecutionResolution]:
    """Prefer the input device, then CPU; break ties by catalog declaration order.

    The environment is a snapshot supplied by the execution shell. Dependency names
    denote usable modules, not merely installed distribution names. WSL is a separate
    platform and requires explicit support in a variant's metadata.
    """
    if task_type not in operation.task_type:
        compatible = sorted(set(operation.task_type).intersection(compatible_task_types(task_type)))
        if not compatible:
            return None
        task_type = compatible[0]

    requested = DeviceType(data_device.type)
    candidates = (requested,) if requested is DeviceType.cpu else (requested, DeviceType.cpu)
    for device_type in candidates:
        if device_type not in environment.available_devices:
            continue
        for implementation in operation.implementations:
            if multi_output and not implementation.supports_multi_output:
                continue
            if device_type not in implementation.devices:
                continue
            if implementation.platforms and environment.platform not in implementation.platforms:
                continue
            if not set(implementation.requires).issubset(environment.available_dependencies):
                continue
            strategy = implementation.strategy_for_task(task_type)
            if strategy is not None:
                device = data_device if device_type is requested else torch.device('cpu')
                return OperationExecutionResolution(implementation.id, device, strategy)

    return None


def resolve_operation_execution(operation: 'OperationMetaInfo', task_type: TaskTypesEnum,
                                data_device: torch.device, environment: ExecutionEnvironment,
                                multi_output: bool = False) -> OperationExecutionResolution:
    result = find_operation_execution(operation, task_type, data_device, environment, multi_output)
    if result is not None:
        return result
    raise ValueError(f'No available implementation for operation {operation.id!r}, '
                     f'task {task_type.value!r}, device {str(data_device)!r}, '
                     f'platform {environment.platform.value!r}.')
