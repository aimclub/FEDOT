import logging

import pytest
from marshmallow import RAISE, Schema, ValidationError, fields

from fedot.validation.boundaries import (
    _format_error_message,
    load_validated,
    load_validated_with_schema_class,
)
from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.validation.fields import OpenUnitInterval
from fedot.validation.recovery import apply_defaults_for_errors, get_field_default


class SampleSchema(Schema):
    class Meta:
        unknown = RAISE

    name = fields.Str(required=True)
    ratio = OpenUnitInterval(load_default=0.2)


class StrictSchema(Schema):
    class Meta:
        unknown = RAISE

    name = fields.Str(required=True)


def test_invalid_value_with_default_logs_warning_and_recovers(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger('test_validation_recovery')
    context = ValidationContext(logger=logger)

    result = load_validated(SampleSchema(), {'name': 'test', 'ratio': 5.0}, context)

    assert result['ratio'] == 0.2
    assert any('Validation failed for ratio' in record.message for record in caplog.records)


def test_invalid_value_without_default_raises():
    with pytest.raises(FedotValidationError):
        load_validated(SampleSchema(), {'ratio': 5.0})


def test_unknown_field_raises_invalid_keys_error():
    with pytest.raises(FedotInvalidKeysError):
        load_validated(SampleSchema(), {'name': 'test', 'extra': 1})


def test_load_validated_with_schema_class():
    result = load_validated_with_schema_class(
        SampleSchema,
        {'name': 'test', 'ratio': 0.5},
    )
    assert result['ratio'] == 0.5


def test_load_validated_non_dict_input_raises():
    with pytest.raises(FedotValidationError):
        load_validated(SampleSchema(), 'not-a-dict')


def test_get_field_default_returns_load_default():
    schema = SampleSchema()
    assert get_field_default(schema, 'ratio') == 0.2


def test_get_field_default_returns_none_for_unknown_field():
    schema = SampleSchema()
    assert get_field_default(schema, 'missing') is None


def test_get_field_default_returns_none_for_required_without_default():
    schema = StrictSchema()
    assert get_field_default(schema, 'name') is None


def test_apply_defaults_for_errors_non_dict_raises():
    exc = ValidationError({'name': ['Missing data for required field.']})
    with pytest.raises(FedotValidationError):
        apply_defaults_for_errors(exc, StrictSchema(), 'not-a-dict')


def test_apply_defaults_for_errors_no_default_raises():
    exc = ValidationError({'name': ['Missing data for required field.']})
    with pytest.raises(FedotValidationError):
        apply_defaults_for_errors(exc, StrictSchema(), {})


def test_recovery_retry_still_fails_raises():
    class BrokenRetrySchema(Schema):
        class Meta:
            unknown = RAISE

        value = fields.Int(load_default='not-an-int')

    with pytest.raises(FedotValidationError):
        load_validated(BrokenRetrySchema(), {'value': 'bad'})


def test_format_error_message_list():
    assert 'a' in _format_error_message(['a', 'b'])
    assert 'b' in _format_error_message(['a', 'b'])


def test_format_error_message_dict():
    formatted = _format_error_message({'field': ['error']})
    assert 'field' in formatted


def test_format_error_message_scalar():
    assert _format_error_message('single error') == 'single error'
