from typing import Any

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates, validates_schema

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class OptionalServiceFittedStateSchema(Schema):
    class Meta:
        unknown = INCLUDE

    has_plan = fields.Bool(required=True)
    has_handlers = fields.Bool(required=True)
    has_cached_handlers = fields.Bool(load_default=False)

    @validates_schema
    def validate_fitted_state(self, data, **kwargs) -> None:
        has_runtime_handlers = data['has_handlers'] or data.get('has_cached_handlers', False)
        if not data['has_plan'] or not has_runtime_handlers:
            raise ValidationError('Optional preprocessing service is not fitted yet')


class OptionalPreprocessingDataTypeSchema(Schema):
    class Meta:
        unknown = INCLUDE

    data_type = fields.Raw(required=True)

    @validates('data_type')
    def validate_data_type(self, value: Any) -> None:
        from fedot.preprocessing.service.tensor_optional_runtime import (
            TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE,
        )

        if value not in TENSOR_OPTIONAL_RUNTIME_BY_DATA_TYPE:
            raise ValidationError(
                f'Optional preprocessing is not supported for data type {value!r}.'
            )


def validate_optional_service_is_fitted(
    has_plan: bool,
    has_handlers: bool,
    has_cached_handlers: bool = False,
    context: ValidationContext = None,
) -> None:
    load_validated(
        OptionalServiceFittedStateSchema(),
        {
            'has_plan': has_plan,
            'has_handlers': has_handlers,
            'has_cached_handlers': has_cached_handlers,
        },
        context,
        prefix='optional_preprocessing',
    )


def validate_optional_preprocessing_data_type(
    data_type: Any,
    context: ValidationContext = None,
) -> Any:
    validated = load_validated(
        OptionalPreprocessingDataTypeSchema(),
        {'data_type': data_type},
        context,
        prefix='optional_preprocessing',
    )
    return validated['data_type']
