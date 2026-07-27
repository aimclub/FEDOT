from typing import Any

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext

SUPPORTED_CLASSIFICATION_OUTPUT_MODES = ('labels', 'probs', 'full_probs', 'default')


class ClassificationOutputModeSchema(Schema):
    class Meta:
        unknown = INCLUDE

    output_mode = fields.Raw(required=True)

    @validates('output_mode')
    def validate_output_mode(self, value: Any) -> None:
        if value is False or value in SUPPORTED_CLASSIFICATION_OUTPUT_MODES:
            return
        raise ValidationError(f'Output model {value} is not supported')


def validate_classification_output_mode(
    output_mode: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        ClassificationOutputModeSchema(),
        {'output_mode': output_mode},
        context,
        prefix='classification',
    )
    return validated['output_mode']
