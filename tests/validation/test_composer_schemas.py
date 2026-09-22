import pytest

from fedot.core.composer.schemas import validate_parallelization_mode
from fedot.validation.errors import FedotValidationError


def test_validate_parallelization_mode_accepts_supported_values():
    assert validate_parallelization_mode('populational') == 'populational'
    assert validate_parallelization_mode('sequential') == 'sequential'


def test_validate_parallelization_mode_rejects_unknown_value():
    with pytest.raises(FedotValidationError, match='Unknown parallelization_mode'):
        validate_parallelization_mode('unknown')
