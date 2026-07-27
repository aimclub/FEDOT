import pytest

from fedot.core.operations.schemas import validate_classification_output_mode
from fedot.validation.errors import FedotValidationError


def test_validate_classification_output_mode_accepts_supported_values():
    assert validate_classification_output_mode('labels') == 'labels'
    assert validate_classification_output_mode(False) is False


def test_validate_classification_output_mode_rejects_unknown_value():
    with pytest.raises(FedotValidationError, match='Output model'):
        validate_classification_output_mode('unknown')
