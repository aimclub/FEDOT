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
    """The "default-or-raise" policy: a field that fails validation but has a
    schema default must be silently replaced with that default and a WARNING
    must be emitted through the context logger.

    Desired behavior: ``ratio=5.0`` violates ``OpenUnitInterval`` (must be in
    (0, 1)), so the loader recovers it to the schema default ``0.2`` and logs
    the failure. The load must *succeed* (not raise), the recovered value must
    be the default, and the warning must mention the offending field name so
    users can diagnose why their input was overridden.
    """
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger('test_validation_recovery')
    context = ValidationContext(logger=logger)

    result = load_validated(SampleSchema(), {'name': 'test', 'ratio': 5.0}, context)

    assert result['ratio'] == 0.2
    assert any('Validation failed for ratio' in record.message for record in caplog.records)


def test_invalid_value_without_default_raises():
    """Recovery only applies when the failing field has a schema default.

    Desired behavior: here ``ratio`` is also invalid (5.0 > 1), but the *real*
    reason this must raise is the missing required ``name`` field, which has no
    default to recover to. A failed field with no default is unrecoverable, so
    the loader raises ``FedotValidationError`` rather than silently inventing a
    value. This is the safeguard that prevents recovery from masking genuine
    user errors.
    """
    with pytest.raises(FedotValidationError):
        load_validated(SampleSchema(), {'ratio': 5.0})


def test_unknown_field_raises_invalid_keys_error():
    """Unknown keys are treated differently from invalid values.

    Desired behavior: an unrecognized field (``extra``) must raise the more
    specific ``FedotInvalidKeysError`` (subclass of ``FedotValidationError``),
    and must never be recovered to a default. Typos in config keys should fail
    loudly and identify the offending keys, since silently dropping them would
    hide user mistakes that almost always indicate a typo.
    """
    with pytest.raises(FedotInvalidKeysError):
        load_validated(SampleSchema(), {'name': 'test', 'extra': 1})


def test_load_validated_with_schema_class():
    """Convenience wrapper instantiates the schema class and delegates to
    ``load_validated``.

    Desired behavior: passing the schema *class* (not an instance) must produce
    the same result as instantiating it manually. This is the only difference
    between the two boundary entry points; everything else (recovery, error
    conversion) must behave identically.
    """
    result = load_validated_with_schema_class(
        SampleSchema,
        {'name': 'test', 'ratio': 0.5},
    )
    assert result['ratio'] == 0.5


def test_load_validated_non_dict_input_raises():
    """Non-dict input is rejected before any field is processed.

    Desired behavior: marshmallow cannot load a scalar/string, so the boundary
    must convert that failure into a ``FedotValidationError`` rather than let
    the raw marshmallow exception escape. This keeps the error type uniform
    across all failure modes for callers that only catch ``FedotValidationError``.
    """
    with pytest.raises(FedotValidationError):
        load_validated(SampleSchema(), 'not-a-dict')


def test_get_field_default_returns_load_default():
    """Recovery introspects ``load_default`` to decide what to substitute.

    Desired behavior: for a field with an explicit ``load_default``, the
    default must be returned as-is. This is the value that recovery patches in
    when the user-supplied value fails, so a wrong return here would silently
    inject the wrong fallback.
    """
    schema = SampleSchema()
    assert get_field_default(schema, 'ratio') == 0.2


def test_get_field_default_returns_none_for_unknown_field():
    """An unknown field has no default to recover to.

    Desired behavior: looking up a field that does not exist on the schema must
    return ``None``. Recovery treats ``None`` as "no default available", which
    forces a raise rather than a silent recovery.
    """
    schema = SampleSchema()
    assert get_field_default(schema, 'missing') is None


def test_get_field_default_returns_none_for_required_without_default():
    """A required field with no default is explicitly unrecoverable.

    Desired behavior: a required field (e.g. ``name``) carries no
    ``load_default``, so ``get_field_default`` returns ``None``. This signals
    recovery that the field cannot be filled in, which is why missing required
    fields always raise regardless of the recovery policy.
    """
    schema = StrictSchema()
    assert get_field_default(schema, 'name') is None


def test_apply_defaults_for_errors_non_dict_raises():
    """Recovery requires a dict to patch; anything else is a caller bug.

    Desired behavior: ``apply_defaults_for_errors`` operates by mutating the
    input dict to fill in recovered defaults, so a non-dict input is not a
    recoverable validation failure but a programming error. It must raise
    ``FedotValidationError`` rather than crash with an unrelated
    ``AttributeError``.
    """
    exc = ValidationError({'name': ['Missing data for required field.']})
    with pytest.raises(FedotValidationError):
        apply_defaults_for_errors(exc, StrictSchema(), 'not-a-dict')


def test_apply_defaults_for_errors_no_default_raises():
    """A failed field that has no default must surface as an error.

    Desired behavior: ``name`` is required and has no default, so recovery
    cannot patch it and must raise ``FedotValidationError``. This is the
    contract that makes recovery safe: it only ever substitutes defaults, never
    invents values, and fails loudly when recovery is impossible.
    """
    exc = ValidationError({'name': ['Missing data for required field.']})
    with pytest.raises(FedotValidationError):
        apply_defaults_for_errors(exc, StrictSchema(), {})


def test_recovery_retry_still_fails_raises():
    """Recovery must not loop forever when the default itself is invalid.

    Desired behavior: the field's ``load_default`` ('not-an-int') is itself
    invalid for an ``Int`` field. After recovery patches in the default and
    re-loads, the second load must still fail, and the loader must raise
    ``FedotValidationError`` instead of retrying indefinitely. This guards
    against a broken schema default turning into an infinite recovery loop.
    """
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
