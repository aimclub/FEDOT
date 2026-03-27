import pytest

from fedot.core.data.data_compatibility_rules import (
    autodetect_tensor_data_type,
    build_data_type_compatibility,
    to_input_compatible_data_type,
    to_tensor_canonical_data_type,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
@pytest.mark.parametrize(
    'raw_data_type, expected',
    [
        (DataTypesEnum.table, DataTypesEnum.tabular),
        (DataTypesEnum.text, DataTypesEnum.tabular),
        (DataTypesEnum.image, DataTypesEnum.ts),
        (DataTypesEnum.multi_ts, DataTypesEnum.ts),
        ('table', DataTypesEnum.tabular),
        ('image', DataTypesEnum.ts),
    ],
)
def test_to_tensor_canonical_data_type_maps_legacy_values(raw_data_type, expected):
    assert to_tensor_canonical_data_type(raw_data_type) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    'raw_data_type, expected',
    [
        (DataTypesEnum.tabular, DataTypesEnum.table),
        (DataTypesEnum.table, DataTypesEnum.table),
        (DataTypesEnum.text, DataTypesEnum.text),
        (DataTypesEnum.ts, DataTypesEnum.ts),
        (DataTypesEnum.image, DataTypesEnum.image),
        ('tabular', DataTypesEnum.table),
        ('multi_time_series', DataTypesEnum.multi_ts),
    ],
)
def test_to_input_compatible_data_type_preserves_legacy_surface(raw_data_type, expected):
    assert to_input_compatible_data_type(raw_data_type) == expected


@pytest.mark.unit
def test_build_data_type_compatibility_contains_both_views():
    compatibility = build_data_type_compatibility(DataTypesEnum.image)

    assert compatibility.original == DataTypesEnum.image
    assert compatibility.tensor_canonical == DataTypesEnum.ts
    assert compatibility.input_compatible == DataTypesEnum.image


@pytest.mark.unit
def test_autodetect_tensor_data_type_prefers_ts_for_forecasting_task():
    assert autodetect_tensor_data_type(Task(TaskTypesEnum.ts_forecasting)) == DataTypesEnum.ts
    assert autodetect_tensor_data_type(Task(TaskTypesEnum.classification)) == DataTypesEnum.tabular
