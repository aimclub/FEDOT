from dataclasses import dataclass
from typing import Iterable, Tuple

from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class PipelineOperationsByRole:
    primary: Tuple[str, ...]
    secondary: Tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            'primary': list(self.primary),
            'secondary': list(self.secondary),
        }


def filter_available_pipeline_operations(preset_operations: Iterable[str],
                                         available_operations: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(preset_operations).intersection(available_operations)))


def build_pipeline_operations_by_role(available_operations: Iterable[str],
                                      task_type: TaskTypesEnum,
                                      ts_data_operations: Iterable[str] = (),
                                      ts_primary_models: Iterable[str] = ()) -> PipelineOperationsByRole:
    normalized_available_operations = tuple(sorted(set(available_operations)))

    if task_type is not TaskTypesEnum.ts_forecasting:
        return PipelineOperationsByRole(
            primary=normalized_available_operations,
            secondary=normalized_available_operations,
        )

    ts_primary_operations = set(ts_data_operations).union(ts_primary_models)
    primary_operations = tuple(
        sorted(ts_primary_operations.intersection(normalized_available_operations)))
    return PipelineOperationsByRole(
        primary=primary_operations,
        secondary=normalized_available_operations,
    )
