from typing import Any

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates

from fedot.core.operations.rules import (
    SUPPORTED_OPTIONAL_IMPUTATION_METHOD_NAMES,
    SUPPORTED_OPTIONAL_SCALING_METHOD_NAMES,
    normalize_optional_method_name,
)
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


class OptionalImputationMethodSchema(Schema):
    class Meta:
        unknown = INCLUDE

    imputation_method = fields.Raw(required=True)

    @validates('imputation_method')
    def validate_imputation_method(self, value: Any) -> None:
        method_name = normalize_optional_method_name(value)
        if method_name not in SUPPORTED_OPTIONAL_IMPUTATION_METHOD_NAMES:
            raise ValidationError(
                f'Unsupported imputation_method for optional preprocessing: {value!r}'
            )


class OptionalScalingMethodSchema(Schema):
    class Meta:
        unknown = INCLUDE

    scaling_method = fields.Raw(required=True)

    @validates('scaling_method')
    def validate_scaling_method(self, value: Any) -> None:
        method_name = normalize_optional_method_name(value)
        if method_name not in SUPPORTED_OPTIONAL_SCALING_METHOD_NAMES:
            raise ValidationError(
                f'Unsupported scaling_method for optional preprocessing: {value!r}'
            )


def validate_optional_imputation_method(
    imputation_method: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        OptionalImputationMethodSchema(),
        {'imputation_method': imputation_method},
        context,
        prefix='optional_preprocessing',
    )
    return validated['imputation_method']


def validate_optional_scaling_method(
    scaling_method: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        OptionalScalingMethodSchema(),
        {'scaling_method': scaling_method},
        context,
        prefix='optional_preprocessing',
    )
    return validated['scaling_method']
