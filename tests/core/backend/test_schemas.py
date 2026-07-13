import pytest

from fedot.core.backend.schemas import validate_backend_name
from fedot.validation.errors import FedotValidationError


def test_validate_backend_name_accepts_cuda_device():
    assert validate_backend_name(' CUDA:3 ') == 'cuda:3'


def test_validate_backend_name_accepts_mps():
    assert validate_backend_name('mps') == 'mps'


def test_validate_backend_name_rejects_unknown_name():
    with pytest.raises(FedotValidationError, match='Unsupported backend name'):
        validate_backend_name('tpu')


def test_validate_backend_name_rejects_empty_string():
    with pytest.raises(FedotValidationError, match='must be a non-empty string'):
        validate_backend_name(' ')
