import pytest

from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.validation.schemas.api_params import (
    validate_api_param_keys,
    validate_problem,
    validate_timeout_generations,
)


def test_validate_problem_accepts_supported_task():
    validate_problem('classification')


def test_validate_problem_rejects_unknown_task():
    with pytest.raises(FedotValidationError, match='Wrong type name'):
        validate_problem('unknown')


def test_validate_timeout_generations_infinite_timeout_with_generations():
    result = validate_timeout_generations(None, 100)
    assert result['num_of_generations'] == 100


def test_validate_timeout_generations_infinite_timeout_without_generations_raises():
    with pytest.raises(FedotValidationError, match='num_of_generations'):
        validate_timeout_generations(None, None)


def test_validate_timeout_generations_invalid_timeout_raises():
    with pytest.raises(FedotValidationError, match='invalid "timeout"'):
        validate_timeout_generations(-5, 10)


def test_validate_api_param_keys_accepts_known_keys():
    validate_api_param_keys({'known': 1}, {'known'})


def test_validate_api_param_keys_rejects_unknown_keys():
    with pytest.raises(FedotInvalidKeysError, match='Invalid key parameters'):
        validate_api_param_keys({'bad': 1}, {'known'})
