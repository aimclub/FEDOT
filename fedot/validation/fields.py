from typing import Any

from marshmallow import ValidationError, fields

class NonEmptyStr(fields.Str):
    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError('must be a non-empty string. Got: {value}')


class OpenUnitInterval(fields.Float):
    """Float in range (0, 1)."""

    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if not 0 < value < 1:
            raise ValidationError('must be in range (0, 1).')


class OpenClosedUnitInterval(fields.Float):
    """Float in range (0, 1]."""

    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if not 0 < value <= 1:
            raise ValidationError('must be in range (0, 1].')


class PositiveFloat(fields.Float):
    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if value <= 0:
            raise ValidationError('must be > 0.')


class NonNegativeFloat(fields.Float):
    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if value < 0:
            raise ValidationError('must be >= 0.')


class PositiveInt(fields.Int):
    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if value <= 0:
            raise ValidationError('must be > 0.')
