"""Execution environment discovery and state carried by a fitted estimator."""
from dataclasses import dataclass
from importlib import import_module
import platform
from typing import Sequence

import torch

from fedot.core.operations.operation_execution_rules import (
    ExecutionEnvironment, OperationExecutionResolution, find_operation_execution,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.operation_implementation import DeviceType, ExecutionPlatform, OperationImplementationInfo


@dataclass(frozen=True)
class FittedOperationExecution:
    resolution: OperationExecutionResolution
    params: OperationParameters


def prepare_device_params(params: OperationParameters, device: torch.device) -> OperationParameters:
    """Bind a library's ordinary device parameter to the selected execution."""
    requested = params.get('device', 'auto')
    if requested not in (None, 'auto'):
        requested = torch.device(requested)
        if requested.type != device.type or (requested.index is not None and requested.index != device.index):
            raise ValueError(f'Model device {str(requested)!r} conflicts with selected execution {str(device)!r}. '
                             'Move TensorData to the desired device before fitting.')
    params.update(device=str(device))
    return params


def filter_available_operations(operations, task_type, data_device: torch.device, multi_output: bool = False):
    """Filter a search pool without changing its presets or creating estimators."""
    implementations = [variant for operation in operations for variant in operation.implementations]
    environment = discover_execution_environment(implementations, data_device)
    return tuple(operation for operation in operations if not operation.implementations or
                 find_operation_execution(operation, task_type, data_device, environment, multi_output) is not None)


def discover_execution_environment(implementations: Sequence[OperationImplementationInfo],
                                   data_device: torch.device) -> ExecutionEnvironment:
    """Probe only dependencies eligible for this platform and the existing tensor device.

    A real TensorData already owns storage on its device; no second hardware choice
    is necessary. Import failures make optional variants unavailable before fitting.
    Other initialization errors remain visible rather than disguising a broken setup.
    """
    system = platform.system().lower()
    release = platform.release().lower()
    execution_platform = (ExecutionPlatform.wsl if system == 'linux' and ('microsoft' in release or 'wsl' in release)
                          else ExecutionPlatform(system))
    devices = frozenset((DeviceType.cpu, DeviceType(data_device.type)))
    dependencies = set()
    required = {dependency for implementation in implementations
                if (not implementation.platforms or execution_platform in implementation.platforms)
                and devices.intersection(implementation.devices)
                for dependency in implementation.requires}
    for dependency in sorted(required):
        try:
            import_module(dependency)
        except ImportError:
            continue
        dependencies.add(dependency)
    return ExecutionEnvironment(execution_platform, devices, frozenset(dependencies))
