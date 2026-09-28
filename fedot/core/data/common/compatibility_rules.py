from dataclasses import dataclass
from typing import Optional, Union

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

TENSOR_CANONICAL_MAPPING = {
    DataTypesEnum.tabular: DataTypesEnum.tabular,
    DataTypesEnum.table: DataTypesEnum.tabular,
    DataTypesEnum.text: DataTypesEnum.tabular,
    DataTypesEnum.ts: DataTypesEnum.ts,
    DataTypesEnum.multi_ts: DataTypesEnum.ts,
    DataTypesEnum.image: DataTypesEnum.ts,
    'tabular': DataTypesEnum.tabular,
    'table': DataTypesEnum.tabular,
    'text': DataTypesEnum.tabular,
    'ts': DataTypesEnum.ts,
    'time_series': DataTypesEnum.ts,
    'multi_ts': DataTypesEnum.ts,
    'multi_time_series': DataTypesEnum.ts,
    'image': DataTypesEnum.ts,
}

INPUT_COMPATIBILITY_MAPPING = {
    DataTypesEnum.tabular: DataTypesEnum.table,
    DataTypesEnum.table: DataTypesEnum.table,
    DataTypesEnum.text: DataTypesEnum.text,
    DataTypesEnum.ts: DataTypesEnum.ts,
    DataTypesEnum.multi_ts: DataTypesEnum.multi_ts,
    DataTypesEnum.image: DataTypesEnum.image,
    'tabular': DataTypesEnum.table,
    'table': DataTypesEnum.table,
    'text': DataTypesEnum.text,
    'ts': DataTypesEnum.ts,
    'time_series': DataTypesEnum.ts,
    'multi_ts': DataTypesEnum.multi_ts,
    'multi_time_series': DataTypesEnum.multi_ts,
    'image': DataTypesEnum.image,
}


@dataclass(frozen=True)
class DataTypeCompatibility:
    original: Union[DataTypesEnum, str]
    tensor_canonical: DataTypesEnum
    input_compatible: DataTypesEnum


def to_tensor_canonical_data_type(data_type: Union[DataTypesEnum, str]) -> DataTypesEnum:
    try:
        return TENSOR_CANONICAL_MAPPING[data_type]
    except KeyError as exc:
        raise ValueError(
            f'Unsupported data_type for tensor compatibility: {data_type}') from exc


def to_input_compatible_data_type(data_type: Union[DataTypesEnum, str]) -> DataTypesEnum:
    try:
        return INPUT_COMPATIBILITY_MAPPING[data_type]
    except KeyError as exc:
        raise ValueError(
            f'Unsupported data_type for input compatibility: {data_type}') from exc


def build_data_type_compatibility(data_type: Union[DataTypesEnum, str]) -> DataTypeCompatibility:
    return DataTypeCompatibility(
        original=data_type,
        tensor_canonical=to_tensor_canonical_data_type(data_type),
        input_compatible=to_input_compatible_data_type(data_type),
    )


def autodetect_tensor_data_type(task: Optional[Task]) -> DataTypesEnum:
    if task is not None and task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    return DataTypesEnum.tabular
