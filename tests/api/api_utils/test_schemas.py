import pytest

from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.api.api_utils.schemas import (
    validate_api_param_keys,
    validate_problem,
    validate_timeout_generations,
)


def test_validate_problem_accepts_supported_task():
    """A known task type must pass without raising."""
    validate_problem('classification')


def test_validate_problem_rejects_unknown_task():
    """An unsupported task type must raise with a 'Wrong type name' message.

    Desired behavior: the schema accepts only regression/classification/ts_forecasting.
    Anything else (e.g. 'unknown') must raise ``FedotValidationError`` with a clear
    message naming the bad value so the user knows which task type is invalid.
    """
    with pytest.raises(FedotValidationError, match='Wrong type name'):
        validate_problem('unknown')


def test_validate_timeout_generations_infinite_timeout_with_generations():
    """When timeout is infinite (None or -1), num_of_generations must be set.

    Desired behavior: ``timeout=None`` means "run until done", which is only safe
    when the user has explicitly set a generation budget. Passing both must
    succeed and return the generations value in the output dict.
    """
    result = validate_timeout_generations(None, 100)
    assert result['num_of_generations'] == 100


def test_validate_timeout_generations_infinite_timeout_without_generations_raises():
    """Infinite timeout without a generation budget is ambiguous and must fail.

    Desired behavior: when ``timeout=None`` and ``num_of_generations=None``, the
    composer would run indefinitely. The validator must catch this and raise
    ``FedotValidationError`` mentioning ``num_of_generations`` so the user knows
    they need to specify a budget.
    """
    with pytest.raises(FedotValidationError, match='num_of_generations'):
        validate_timeout_generations(None, None)


def test_validate_timeout_generations_invalid_timeout_raises():
    """A negative timeout (other than the sentinel -1) must be rejected.

    Desired behavior: ``timeout=-5`` is neither a positive duration nor the
    "infinite" sentinel, so it must raise with 'invalid "timeout"' in the
    message. This prevents nonsensical configs from reaching the composer.
    """
    with pytest.raises(FedotValidationError, match='invalid "timeout"'):
        validate_timeout_generations(-5, 10)


def test_validate_api_param_keys_accepts_known_keys():
    """Keys present in the allowed set must pass without raising."""
    validate_api_param_keys({'known': 1}, {'known'})


def test_validate_api_param_keys_rejects_unknown_keys():
    """A key not in the allowed set must raise FedotInvalidKeysError.

    Desired behavior: ``build_api_params_keys_schema`` dynamically builds a
    RAISE-mode schema from the allowed key set. Any key outside that set (e.g.
    'bad') must surface as ``FedotInvalidKeysError`` with 'Invalid key
    parameters' in the message, so users see a typo-friendly error rather than
    the unknown key being silently ignored.
    """
    with pytest.raises(FedotInvalidKeysError, match='Invalid key parameters'):
        validate_api_param_keys({'bad': 1}, {'known'})
