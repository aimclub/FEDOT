from typing import Optional

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext

SUPPORTED_PARALLELIZATION_MODES = ('populational', 'sequential')


class ParallelizationModeSchema(Schema):
    class Meta:
        unknown = INCLUDE

    parallelization_mode = fields.Str(required=True)

    @validates('parallelization_mode')
    def validate_parallelization_mode(self, value: str) -> None:
        if value not in SUPPORTED_PARALLELIZATION_MODES:
            raise ValidationError(f'Unknown parallelization_mode: {value}')


def validate_parallelization_mode(
    parallelization_mode: str,
    context: ValidationContext = None,
) -> str:
    validated = load_validated(
        ParallelizationModeSchema(),
        {'parallelization_mode': parallelization_mode},
        context,
        prefix='composer_requirements',
    )
    return validated['parallelization_mode']
