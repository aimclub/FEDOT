from typing import Any, Optional, Type

from marshmallow import Schema, ValidationError

from fedot.validation.context import ValidationContext
from fedot.validation.errors import FedotValidationError, from_marshmallow_error
from fedot.validation.recovery import apply_defaults_for_errors


def load_validated(
    schema: Schema,
    data: Any,
    context: Optional[ValidationContext] = None,
    *,
    prefix: str = '',
) -> Any:
    """Load data through a Marshmallow schema with default-or-raise policy."""
    validation_context = context or ValidationContext()
    schema.context['fedot_validation_context'] = validation_context

    try:
        return schema.load(data)
    except ValidationError as exc:
        if not isinstance(data, dict):
            raise from_marshmallow_error(exc, prefix=prefix) from exc

        try:
            patched_data, applied_defaults = apply_defaults_for_errors(
                exc, schema, data, prefix=prefix)
        except FedotValidationError:
            raise
        except ValidationError as recovery_exc:
            raise from_marshmallow_error(recovery_exc, prefix=prefix) from recovery_exc

        if not applied_defaults:
            raise from_marshmallow_error(exc, prefix=prefix) from exc

        logger = validation_context.get_logger()
        for field_path, error_message, default_value in applied_defaults:
            logger.warning(
                'Validation failed for %s: %s. Using default %r.',
                field_path,
                _format_error_message(error_message),
                default_value,
            )

        try:
            return schema.load(patched_data)
        except ValidationError as retry_exc:
            raise from_marshmallow_error(retry_exc, prefix=prefix) from retry_exc


def load_validated_with_schema_class(
    schema_cls: Type[Schema],
    data: Any,
    context: Optional[ValidationContext] = None,
    *,
    prefix: str = '',
) -> Any:
    return load_validated(schema_cls(), data, context, prefix=prefix)


def _format_error_message(error_message: Any) -> str:
    if isinstance(error_message, list):
        return '; '.join(str(item) for item in error_message)
    if isinstance(error_message, dict):
        return str(error_message)
    return str(error_message)
