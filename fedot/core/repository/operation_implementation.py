"""Implementation metadata without importing optional model libraries."""
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Type

from fedot.core.repository.json_evaluation import import_enums_from_str, import_strategy_from_str
from fedot.core.repository.tasks import TaskTypesEnum

if TYPE_CHECKING:
    from fedot.core.operations.strategies.base import EvaluationStrategy


class DeviceType(Enum):
    cpu = 'cpu'
    cuda = 'cuda'
    mps = 'mps'


class ExecutionPlatform(Enum):
    """WSL is separate so native-Linux support does not implicitly enable it."""

    linux = 'linux'
    windows = 'windows'
    darwin = 'darwin'
    wsl = 'wsl'


@dataclass(frozen=True)
class StrategyReference:
    module: str
    name: str

    def load(self) -> Type['EvaluationStrategy']:
        """Import the selected strategy at the execution boundary, not catalog loading."""
        return import_strategy_from_str([self.module, self.name])


@dataclass(frozen=True)
class OperationImplementationInfo:
    id: str
    devices: Tuple[DeviceType, ...]
    strategies: Tuple[Tuple[TaskTypesEnum, StrategyReference], ...]
    platforms: Tuple[ExecutionPlatform, ...] = ()
    requires: Tuple[str, ...] = ()
    supports_multi_output: bool = False

    def strategy_for_task(self, task: TaskTypesEnum) -> Optional[StrategyReference]:
        return next((strategy for task_type, strategy in self.strategies if task_type == task), None)


def parse_operation_implementations(properties: Mapping, metadata: Mapping,
                                    task_types: Sequence[TaskTypesEnum]) -> Tuple[OperationImplementationInfo, ...]:
    """Resolve shared metadata and per-variant overrides, preserving declaration order.

    Only explicit variants participate. Operations awaiting migration retain an empty
    tuple, rather than acquiring guessed CPU capabilities from their old strategy.
    """
    implementations = []
    for implementation_id, definition in properties.get('implementations', {}).items():
        shared = metadata[definition['meta']] if 'meta' in definition else {}
        description = {**shared, **definition}
        strategies = description['strategies']
        if isinstance(strategies, list):
            reference = StrategyReference(*strategies)
            references = tuple((task, reference) for task in task_types)
        elif isinstance(strategies, dict):
            references = tuple((task, StrategyReference(*reference))
                               for tasks, reference in strategies.items()
                               for task in import_enums_from_str(tasks))
        else:
            raise TypeError('Implementation strategies must be a reference or a task mapping')

        implementations.append(OperationImplementationInfo(
            id=implementation_id,
            devices=tuple(DeviceType(device) for device in description['devices']),
            strategies=references,
            platforms=tuple(ExecutionPlatform(platform) for platform in description.get('platforms', ())),
            requires=tuple(description.get('requires', ())),
            supports_multi_output=description.get('supports_multi_output', False),
        ))
    return tuple(implementations)
