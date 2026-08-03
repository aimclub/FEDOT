from marshmallow import INCLUDE, Schema, ValidationError, fields, validates_schema

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class OptionalServiceFittedStateSchema(Schema):
    class Meta:
        unknown = INCLUDE

    has_plan = fields.Bool(required=True)
    has_handlers = fields.Bool(required=True)

    @validates_schema
    def validate_fitted_state(self, data, **kwargs) -> None:
        if not data['has_plan'] or not data['has_handlers']:
            raise ValidationError('Optional preprocessing service is not fitted yet')


def validate_optional_service_is_fitted(
    has_plan: bool,
    has_handlers: bool,
    context: ValidationContext = None,
) -> None:
    load_validated(
        OptionalServiceFittedStateSchema(),
        {
            'has_plan': has_plan,
            'has_handlers': has_handlers,
        },
        context,
        prefix='optional_preprocessing',
    )
