from typing import Any, Iterable, Sequence, Tuple

from marshmallow import ValidationError, fields, validate


def validate_sorted_unique_ratios(ratios: Sequence[float]) -> None:
    if not ratios:
        raise ValidationError(
            '"sampling_config.candidate_ratios" must be a non-empty list of floats.')
    normalized = [float(ratio) for ratio in ratios]
    if len(set(normalized)) != len(normalized):
        raise ValidationError(
            '"sampling_config.candidate_ratios" must not contain duplicates.')
    if tuple(sorted(normalized)) != tuple(normalized):
        raise ValidationError(
            '"sampling_config.candidate_ratios" must be sorted in ascending order without duplicates.')


class NonEmptyStr(fields.Str):
    def _validate(self, value: Any) -> None:
        super()._validate(value)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError('must be a non-empty string.')


class SortedUniqueFloatTuple(fields.List):
    def __init__(self, **kwargs):
        super().__init__(fields.Float(), **kwargs)

    def _deserialize(self, value: Any, attr, data, **kwargs) -> Tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValidationError(
                '"sampling_config.candidate_ratios" must be a non-empty list of floats.')
        normalized = []
        for ratio in value:
            if not isinstance(ratio, (float, int)):
                raise ValidationError(
                    '"sampling_config.candidate_ratios" must contain only numbers.')
            ratio = float(ratio)
            if not 0 < ratio <= 1:
                raise ValidationError(
                    '"sampling_config.candidate_ratios" values must be in range (0, 1].')
            normalized.append(ratio)
        validate_sorted_unique_ratios(normalized)
        return tuple(normalized)


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


def one_of_validator(choices: Iterable[Any], *, error_template: str) -> validate.OneOf:
    return validate.OneOf(list(choices), error=error_template)
