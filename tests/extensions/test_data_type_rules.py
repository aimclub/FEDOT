import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.extensions.data_type_rules import build_extension_data_type_view


@pytest.mark.unit
def test_build_extension_data_type_view_normalizes_legacy_and_tensor_types():
    view = build_extension_data_type_view(
        (DataTypesEnum.text, DataTypesEnum.table, DataTypesEnum.image))

    assert view.input_types == (
        DataTypesEnum.text, DataTypesEnum.table, DataTypesEnum.image)
    assert view.tensor_types == (DataTypesEnum.tabular, DataTypesEnum.ts)
    assert view.preferred_output_type_name == DataTypesEnum.text.name


@pytest.mark.unit
def test_build_extension_data_type_view_uses_default_table_when_empty():
    view = build_extension_data_type_view(())

    assert view.input_types == (DataTypesEnum.table,)
    assert view.tensor_types == (DataTypesEnum.tabular,)
    assert view.preferred_output_type_name == DataTypesEnum.table.name
