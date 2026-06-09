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
    exc = ValidationError({'unknown_param': ['Unknown field.']})
    mapped = from_marshmallow_error(exc)
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'unknown_param' in mapped.invalid_keys


def test_from_marshmallow_error_maps_unknown_fields_with_prefix():
    exc = ValidationError({'unknown_param': ['Unknown field.']})
    mapped = from_marshmallow_error(exc, prefix='api_params')
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'api_params' in str(mapped)


def test_from_marshmallow_error_maps_field_errors():
    exc = ValidationError('bad value', field_name='validation_size')
    mapped = from_marshmallow_error(exc, prefix='chunked_ensemble_config')
    assert isinstance(mapped, FedotValidationError)
    assert mapped.field_name == 'chunked_ensemble_config.validation_size'


def test_from_marshmallow_error_reraises_fedot_validation_error():
    original = FedotValidationError('already mapped', field_name='field')
    mapped = from_marshmallow_error(original)
    assert mapped is original


def test_from_marshmallow_error_nested_unknown_keys():
    exc = ValidationError({'nested': {'inner_unknown': ['Unknown field.']}})
    mapped = from_marshmallow_error(exc)
    assert isinstance(mapped, FedotInvalidKeysError)
    assert 'inner_unknown' in mapped.invalid_keys
