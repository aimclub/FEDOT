from typing import Any, Dict, List, Tuple

from marshmallow import Schema, ValidationError
from marshmallow.utils import missing as MISSING

from fedot.validation.errors import from_marshmallow_error


def get_field_default(schema: Schema, field_name: str) -> Any:
    field = schema.fields.get(field_name)
    if field is None:
        return None
    if field.load_default is not MISSING:
        default = field.load_default
        if callable(default):
            return default()
        return default
    return None


def _iter_error_fields(messages: Any, prefix: str = '') -> List[Tuple[str, Any]]:
    if not isinstance(messages, dict):
        return [(prefix or '_schema', messages)]

    result = []
    for key, value in messages.items():
        field_path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.extend(_iter_error_fields(value, field_path))
        else:
            result.append((field_path, value))
    return result


def _is_unknown_field_error(message: Any) -> bool:
    if isinstance(message, list):
        return any(_is_unknown_field_error(item) for item in message)
    return isinstance(message, str) and 'Unknown field' in message


def apply_defaults_for_errors(
    exc: ValidationError,
    schema: Schema,
    data: Dict[str, Any],
    *,
    prefix: str = '',
) -> Tuple[Dict[str, Any], List[Tuple[str, Any, Any]]]:
    """Patch data with schema defaults for failed fields that have defaults.

    Returns patched data and list of (field_path, error_message, default_value).
    Raises FedotValidationError if any failed field has no default.
    """
    if not isinstance(data, dict):
        raise from_marshmallow_error(exc, prefix=prefix)

    if isinstance(exc.messages, dict) and any(
        _is_unknown_field_error(value) for value in exc.messages.values()
    ):
        raise from_marshmallow_error(exc, prefix=prefix)

    patched = dict(data)
    applied: List[Tuple[str, Any, Any]] = []

    for field_path, error_message in _iter_error_fields(exc.messages):
        if _is_unknown_field_error(error_message):
            raise from_marshmallow_error(exc, prefix=prefix)

        top_level_field = field_path.split('.')[0]
        default_value = get_field_default(schema, top_level_field)
        if default_value is None:
            raise from_marshmallow_error(exc, prefix=prefix)
        patched[top_level_field] = default_value
        applied.append((field_path, error_message, default_value))

    return patched, applied
