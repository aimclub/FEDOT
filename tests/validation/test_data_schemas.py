import pytest

from fedot.core.data.schemas import (
    validate_tabular_file_path,
    validate_tensor_data_merge_data_type,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.validation.errors import FedotValidationError


def test_validate_tabular_file_path_accepts_existing_csv(tmp_path):
    csv_file = tmp_path / 'data.csv'
    csv_file.write_text('a,b\n1,2\n', encoding='utf-8')

    result = validate_tabular_file_path(str(csv_file))

    assert result == str(csv_file)


def test_validate_tabular_file_path_rejects_unsupported_extension(tmp_path):
    json_file = tmp_path / 'data.json'
    json_file.write_text('{}', encoding='utf-8')

    with pytest.raises(FedotValidationError, match='Unsupported tabular file format'):
        validate_tabular_file_path(str(json_file))


def test_validate_tabular_file_path_rejects_missing_file(tmp_path):
    missing = tmp_path / 'missing.csv'

    with pytest.raises(FedotValidationError, match='does not exist'):
        validate_tabular_file_path(str(missing))


def test_validate_tensor_data_merge_data_type_accepts_data_type():
    assert validate_tensor_data_merge_data_type(DataTypesEnum.tabular) is DataTypesEnum.tabular


def test_validate_tensor_data_merge_data_type_rejects_none():
    with pytest.raises(FedotValidationError, match="Can't merge different TensorData data types"):
        validate_tensor_data_merge_data_type(None)
