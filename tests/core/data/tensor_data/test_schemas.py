import pytest

from fedot.validation.errors import FedotValidationError
from fedot.core.data.tensor_data.schemas import validate_tabular_file_path


def test_validate_tabular_file_path_accepts_existing_csv(tmp_path):
    """An existing .csv file must pass and return the path string."""
    csv_file = tmp_path / 'data.csv'
    csv_file.write_text('a,b\n1,2\n', encoding='utf-8')

    result = validate_tabular_file_path(str(csv_file))
    assert result == str(csv_file)


def test_validate_tabular_file_path_rejects_unsupported_extension(tmp_path):
    """Only .csv and .tsv are supported; other extensions must be rejected.

    Desired behavior: a .json file exists on disk but is not a supported
    tabular format. The validator must raise ``FedotValidationError`` with
    'Unsupported tabular file format' rather than attempting to read it, so
    the user is immediately told which formats are accepted.
    """
    json_file = tmp_path / 'data.json'
    json_file.write_text('{}', encoding='utf-8')

    with pytest.raises(FedotValidationError, match='Unsupported tabular file format'):
        validate_tabular_file_path(str(json_file))


def test_validate_tabular_file_path_rejects_missing_file(tmp_path):
    """A path that does not exist on the filesystem must be rejected.

    Desired behavior: the schema checks ``Path.is_file()`` so that users get
    a clear 'does not exist' error at config time, not a cryptic FileNotFoundError
    later when the data loader tries to open the file.
    """
    missing = tmp_path / 'missing.csv'

    with pytest.raises(FedotValidationError, match='does not exist'):
        validate_tabular_file_path(str(missing))
