import pytest
from marshmallow import ValidationError

from fedot.validation.errors import (
    FedotInvalidKeysError,
    FedotValidationError,
    field_error,
    from_marshmallow_error,
    unknown_keys_error,
)


def test_fedot_validation_error_subclasses_marshmallow_validation_error():
    assert issubclass(FedotValidationError, ValidationError)


def test_fedot_invalid_keys_error_subclasses_fedot_validation_error():
    assert issubclass(FedotInvalidKeysError, FedotValidationError)


def test_unknown_keys_error_with_prefix():
    """Factory for unknown-key errors, namespacing the message with a prefix.

    Desired behavior: the prefix (typically the config section, e.g.
    ``sampling_config``) must appear in the rendered message so the user knows
    *where* the unknown key lives, and every offending key must be listed. The
    result must be a ``FedotInvalidKeysError`` so callers can distinguish typo
    errors from value errors.
    """
    error = unknown_keys_error('sampling_config', {'foo', 'bar'})
    assert isinstance(error, FedotInvalidKeysError)
    assert 'sampling_config' in str(error)
    assert 'foo' in str(error)


def test_fedot_invalid_keys_error_without_prefix():
    error = FedotInvalidKeysError({'unknown'})
    assert 'Invalid key parameters' in str(error)
    assert 'unknown' in error.invalid_keys


def test_field_error_sets_field_name():
    error = field_error('sampling_config.provider', 'must be non-empty')
    assert error.field_name == 'sampling_config.provider'
    assert 'must be non-empty' in str(error.messages)


def test_from_marshmallow_error_maps_unknown_fields_to_invalid_keys():
    """Raw "Unknown field" marshmallow errors are reclassified as
    ``FedotInvalidKeysError``.

    Desired behavior: marshmallow signals unknown keys with the literal message
    'Unknown field'. The converter must detect that message and promote the
    error to the more specific ``FedotInvalidKeysError``, collecting the key
    names into ``invalid_keys``. This lets FEDOT code branch on "user typo"
    vs. "bad value" by exception type.
    """
    exc = ValidationError({'unknown_param': ['Unknown field.']})
    mapped = from_marshmallow_error(exc)
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'unknown_param' in mapped.invalid_keys


def test_from_marshmallow_error_maps_unknown_fields_with_prefix():
    """The prefix is prepended to the field path in converted errors.

    Desired behavior: with ``prefix='api_params'``, the rendered message must
    namespace the error under that section so multi-section configs report
    *which* config the unknown key belongs to.
    """
    exc = ValidationError({'unknown_param': ['Unknown field.']})
    mapped = from_marshmallow_error(exc, prefix='api_params')
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'api_params' in str(mapped)


def test_from_marshmallow_error_maps_field_errors():
    """Non-unknown-field errors are wrapped as generic ``FedotValidationError``
    with a dotted ``prefix.field`` path.

    Desired behavior: a value error on ``validation_size`` (not an unknown key)
    must stay a base ``FedotValidationError`` (not ``FedotInvalidKeysError``),
    and the ``field_name`` must be joined as
    ``chunked_ensemble_config.validation_size`` so the path the user sees
    matches the config hierarchy.
    """
    exc = ValidationError('bad value', field_name='validation_size')
    mapped = from_marshmallow_error(exc, prefix='chunked_ensemble_config')
    assert isinstance(mapped, FedotValidationError)
    assert mapped.field_name == 'chunked_ensemble_config.validation_size'


def test_from_marshmallow_error_reraises_fedot_validation_error():
    """Already-converted errors must pass through untouched (idempotency).

    Desired behavior: ``from_marshmallow_error`` may be called more than once
    on the same exception (e.g. nested recovery retries). If the input is
    already a ``FedotValidationError``, it must be returned as-is so the
    prefix/path is not double-applied and the original instance identity is
    preserved.
    """
    original = FedotValidationError('already mapped', field_name='field')
    mapped = from_marshmallow_error(original)
    assert mapped is original


def test_from_marshmallow_error_nested_unknown_keys():
    """Unknown-key detection must recurse into nested error dicts.

    Desired behavior: marshmallow nests errors for nested schemas, so an
    'Unknown field' can sit under a parent key. The converter must walk into
    nested dicts and still collect the inner key names into ``invalid_keys``,
    otherwise typos inside nested config sections would be misclassified as
    generic value errors.
    """
    exc = ValidationError({'nested': {'inner_unknown': ['Unknown field.']}})
    mapped = from_marshmallow_error(exc)
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'inner_unknown' in mapped.invalid_keys
