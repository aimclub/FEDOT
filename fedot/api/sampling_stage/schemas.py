from typing import Any, Dict, Sequence, Tuple

from marshmallow import RAISE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.fields import (
    NonEmptyStr,
    NonNegativeFloat,
    OpenClosedUnitInterval,
    OpenUnitInterval,
    PositiveFloat,
)


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


class SamplingConfigBaseSchema(Schema):
    class Meta:
        unknown = RAISE

    strategy_kind = fields.Str(required=True)
    provider = NonEmptyStr(load_default='sampling_zoo')
    strategy = NonEmptyStr(load_default='random')
    strategy_params = fields.Dict(keys=fields.Str(), values=fields.Raw(), load_default=lambda: {})
    cap_max_timeout_share = OpenClosedUnitInterval(load_default=0.35)
    min_automl_time_minutes = PositiveFloat(load_default=0.1)
    infinite_timeout_cap_minutes = PositiveFloat(load_default=5.0)
    random_state = fields.Int(allow_none=True, load_default=42)

    @validates('strategy_kind')
    def validate_strategy_kind(self, value: str) -> None:
        if value not in ('subset', 'chunking'):
            raise ValidationError(
                '"sampling_config.strategy_kind" must be "subset" or "chunking".')

    @validates('provider')
    def validate_provider(self, value: str) -> None:
        if not value.strip():
            raise ValidationError('"sampling_config.provider" must be a non-empty string.')

    @validates('strategy')
    def validate_strategy(self, value: str) -> None:
        if not value.strip():
            raise ValidationError('"sampling_config.strategy" must be a non-empty string.')

    @validates('strategy_params')
    def validate_strategy_params(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValidationError('"sampling_config.strategy_params" must be a dictionary.')

    @validates('cap_max_timeout_share')
    def validate_cap_max_timeout_share(self, value: float) -> None:
        if not 0 < value <= 1:
            raise ValidationError(
                '"sampling_config.cap_max_timeout_share" must be in range (0, 1].')

    @validates('min_automl_time_minutes')
    def validate_min_automl_time_minutes(self, value: float) -> None:
        if value <= 0:
            raise ValidationError('"sampling_config.min_automl_time_minutes" must be > 0.')

    @validates('infinite_timeout_cap_minutes')
    def validate_infinite_timeout_cap_minutes(self, value: float) -> None:
        if value <= 0:
            raise ValidationError('"sampling_config.infinite_timeout_cap_minutes" must be > 0.')

    @validates('random_state')
    def validate_random_state(self, value) -> None:
        if value is not None and not isinstance(value, int):
            raise ValidationError('"sampling_config.random_state" must be int or None.')


class SamplingSubsetConfigSchema(SamplingConfigBaseSchema):
    candidate_ratios = SortedUniqueFloatTuple(load_default=(0.15, 0.2, 0.3, 0.5, 0.7))
    delta_metric_threshold = NonNegativeFloat(load_default=0.03)
    validation_size = OpenUnitInterval(load_default=0.2)

    @validates('delta_metric_threshold')
    def validate_delta_metric_threshold(self, value: float) -> None:
        if value < 0:
            raise ValidationError('"sampling_config.delta_metric_threshold" must be >= 0.')

    @validates('validation_size')
    def validate_validation_size(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValidationError('"sampling_config.validation_size" must be in range (0, 1).')


class SamplingChunkingConfigSchema(SamplingConfigBaseSchema):
    pass


class SamplingConfigSchema(Schema):
    """Discriminated union loader for sampling configuration."""

    def load(self, data, *, many=None, partial=None, unknown=None, **kwargs) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValidationError('"sampling_config" must be a dictionary or None.')

        validation_context = self.context.get('fedot_validation_context', ValidationContext())

        strategy_kind = data.get('strategy_kind')
        if strategy_kind is None:
            raise ValidationError(
                '"sampling_config.strategy_kind" must be provided.',
                field_name='strategy_kind')

        if strategy_kind == 'subset':
            return load_validated(
                SamplingSubsetConfigSchema(),
                data,
                validation_context,
                prefix='sampling_config',
            )
        if strategy_kind == 'chunking':
            return load_validated(
                SamplingChunkingConfigSchema(),
                data,
                validation_context,
                prefix='sampling_config',
            )
        raise ValidationError(
            '"sampling_config.strategy_kind" must be "subset" or "chunking".',
            field_name='strategy_kind')
