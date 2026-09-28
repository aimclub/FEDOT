from typing import Any, Optional, Sequence

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates, validates_schema

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


class OptionalServicePredictReadySchema(Schema):
    """Validate that optional service can run predict for its plan steps."""

    class Meta:
        unknown = INCLUDE

    plan = fields.Raw(required=True, allow_none=True)
    fitted_handlers = fields.Raw(required=True, allow_none=True)
    cached_handler_paths = fields.Raw(load_default=tuple)

    @validates_schema
    def validate_predict_ready(self, data, **kwargs) -> None:
        plan = data['plan']
        if plan is None:
            raise ValidationError('Optional preprocessing service is not fitted yet')

        steps = getattr(plan, 'steps', None)
        if steps is None:
            raise ValidationError('Optional preprocessing service is not fitted yet')

        n_steps = len(steps)
        fitted_handlers = data['fitted_handlers']
        cached_handler_paths = data.get('cached_handler_paths') or ()

        if fitted_handlers is not None:
            n_handlers = len(fitted_handlers)
        else:
            n_handlers = len(cached_handler_paths)

        if n_steps == 0:
            return

        if n_handlers == 0:
            raise ValidationError(
                'All required optional preprocessing handlers must be trained'
            )

        if n_handlers != n_steps:
            raise ValidationError(
                'All required optional preprocessing handlers must be trained '
                f'(plan has {n_steps} steps, got {n_handlers} handlers)'
            )


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


def validate_optional_service_predict_ready(
    plan: Any,
    fitted_handlers: Optional[Sequence[Any]],
    cached_handler_paths: Optional[Sequence[Any]] = None,
    context: ValidationContext = None,
) -> None:
    """Fail-fast if plan/handlers are missing or their lengths do not match."""
    load_validated(
        OptionalServicePredictReadySchema(),
        {
            'plan': plan,
            'fitted_handlers': fitted_handlers,
            'cached_handler_paths': tuple(cached_handler_paths or ()),
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
