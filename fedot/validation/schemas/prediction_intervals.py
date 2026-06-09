from typing import Any, Dict, Union

from marshmallow import RAISE, Schema, ValidationError, fields, validates, validates_schema

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext
from fedot.validation.fields import NonNegativeFloat, PositiveInt


class PredictionIntervalsInitSchema(Schema):
    class Meta:
        unknown = RAISE

    horizon = fields.Int(allow_none=True, load_default=None)
    nominal_error = NonNegativeFloat(load_default=0.1)
    method = fields.Str(load_default='last_generation_ql')

    @validates('horizon')
    def validate_horizon(self, value) -> None:
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValidationError(
                'Argument horizon must be None or positive integer number.')

    @validates('nominal_error')
    def validate_nominal_error(self, value: float) -> None:
        if not (0 <= value <= 1):
            raise ValidationError(
                'Argument nominal_error must be float number between 0 and 1.')

    @validates('method')
    def validate_method(self, value: str) -> None:
        available_methods = (
            'last_generation_ql',
            'best_pipelines_quantiles',
            'mutation_of_best_pipeline',
        )
        if value not in available_methods:
            raise ValidationError(
                "Argument 'method' is incorrect. Possible options: "
                "'last_generation_ql', 'best_pipelines_quantiles', "
                "'mutation_of_best_pipeline'.")


class PredictionIntervalsParamsSchema(Schema):
    class Meta:
        unknown = RAISE

    logging_level = fields.Int(load_default=20)
    n_jobs = fields.Int(load_default=-1)
    show_progress = fields.Bool(load_default=True)
    number_mutations = PositiveInt(load_default=30)
    mutations_choice = fields.Str(load_default='different')
    mutations_discard_inapropriate_pipelines = fields.Bool(load_default=True)
    mutations_keep_percentage = NonNegativeFloat(load_default=0.66)
    mutations_operations = fields.List(fields.Str(), load_default=list)
    ql_number_models = fields.Raw(load_default=10)
    ql_tuner_iterations = PositiveInt(load_default=10)
    ql_tuner_minutes = PositiveInt(load_default=1)
    bpq_number_models = fields.Raw(load_default=10)

    @validates('logging_level')
    def validate_logging_level(self, value: int) -> None:
        if value not in (0, 10, 20, 30, 40, 50):
            raise ValidationError(
                'Argument logging_level must be in [0, 10, 20, 30, 40, 50].')

    @validates('n_jobs')
    def validate_n_jobs(self, value: int) -> None:
        if not isinstance(value, int) or value == 0 or value < -1:
            raise ValidationError(
                'Argument n_jobs must be -1 or positive integer number.')

    @validates('mutations_choice')
    def validate_mutations_choice(self, value: str) -> None:
        if value not in ('different', 'with_replacement'):
            raise ValidationError(
                "Arument mutations_choice is incorrect. Options: 'different' and 'with_replacement'.")

    @validates('mutations_keep_percentage')
    def validate_mutations_keep_percentage(self, value: float) -> None:
        if not (0 <= value <= 1):
            raise ValidationError(
                'Argument mutation_keep_percentage must be float number between 0 and 1.')

    @validates('mutations_operations')
    def validate_mutations_operations(self, value) -> None:
        if not isinstance(value, list):
            raise ValidationError(
                'Argument mutations_operations must be list of strings.')

    @validates('ql_number_models')
    def validate_ql_number_models(self, value: Union[int, str]) -> None:
        if value != 'max' and (not isinstance(value, int) or value < 1):
            raise ValidationError(
                "Argument ql_number_models must be positive integer number or 'max'.")

    @validates('ql_tuner_minutes')
    def validate_ql_tuner_minutes(self, value) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValidationError(
                'Argument ql_tuner_minutes must be positive real number.')

    @validates('bpq_number_models')
    def validate_bpq_number_models(self, value: Union[int, str]) -> None:
        if value != 'max' and (not isinstance(value, int) or value < 1):
            raise ValidationError(
                "Argument bpq_number_models must be positive integer number or 'max'.")


def validate_prediction_intervals_init(
    horizon,
    nominal_error: float,
    method: str,
    params_dict: Dict[str, Any],
    context: ValidationContext = None,
) -> Dict[str, Any]:
    load_validated(
        PredictionIntervalsInitSchema(),
        {'horizon': horizon, 'nominal_error': nominal_error, 'method': method},
        context,
        prefix='prediction_intervals',
    )
    return load_validated(
        PredictionIntervalsParamsSchema(),
        params_dict,
        context,
        prefix='prediction_intervals.params',
    )
