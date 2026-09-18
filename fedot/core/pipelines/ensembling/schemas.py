from marshmallow import RAISE, Schema, ValidationError, fields, validates

from fedot.validation.fields import OpenUnitInterval, PositiveInt

_ENSEMBLE_METHODS = ('voting', 'weighted', 'routed_weighted', 'gated_weighted')


class ChunkedEnsembleConfigSchema(Schema):
    class Meta:
        unknown = RAISE

    validation_size = OpenUnitInterval(load_default=0.2)
    validation_split_seed = fields.Int(allow_none=True, load_default=42)
    ensemble_method = fields.Str(load_default='voting')
    ensemble_params = fields.Dict(keys=fields.Str(), values=fields.Raw(), load_default=lambda: {})
    batch_size = PositiveInt(load_default=10000)
    min_successful_chunks = PositiveInt(load_default=1)

    @validates('validation_size')
    def validate_validation_size(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValidationError(
                '"chunked_ensemble_config.validation_size" must be in range (0, 1).')

    @validates('validation_split_seed')
    def validate_validation_split_seed(self, value) -> None:
        if value is not None and not isinstance(value, int):
            raise ValidationError(
                '"chunked_ensemble_config.validation_split_seed" must be int or None.')

    @validates('ensemble_method')
    def validate_ensemble_method(self, value: str) -> None:
        if value not in _ENSEMBLE_METHODS:
            raise ValidationError(
                '"chunked_ensemble_config.ensemble_method" must be one of '
                '{"voting", "weighted", "routed_weighted", "gated_weighted"}.')

    @validates('ensemble_params')
    def validate_ensemble_params(self, value) -> None:
        if not isinstance(value, dict):
            raise ValidationError(
                '"chunked_ensemble_config.ensemble_params" must be a dictionary.')

    @validates('batch_size')
    def validate_batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValidationError('"chunked_ensemble_config.batch_size" must be > 0.')

    @validates('min_successful_chunks')
    def validate_min_successful_chunks(self, value: int) -> None:
        if value <= 0:
            raise ValidationError(
                '"chunked_ensemble_config.min_successful_chunks" must be > 0.')
