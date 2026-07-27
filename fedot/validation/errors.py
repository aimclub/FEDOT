from typing import Any, FrozenSet, Optional, Set, Union

from marshmallow import ValidationError


class FedotValidationError(ValidationError):
    """Public FEDOT validation error. Subclasses marshmallow.ValidationError."""

    code: str = 'validation_error'

    def __init__(
        self,
        message: Union[str, list, dict],
        field_name: str = '_schema',
        *,
        code: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, field_name=field_name, **kwargs)
        if code is not None:
            self.code = code


class FedotInvalidKeysError(FedotValidationError):
    """Unknown keys in a config or params dict."""

    code = 'invalid_keys'

    def __init__(
        self,
        invalid_keys: Set[str],
        *,
        prefix: str = '',
        message: Optional[str] = None,
    ):
        self.invalid_keys = frozenset(invalid_keys)
        if message is None:
            if prefix:
                message = f'Unknown keys in "{prefix}": {sorted(invalid_keys)}'
            else:
                message = f'Invalid key parameters {invalid_keys}'
        super().__init__(message, field_name='_schema', code=self.code)


def field_error(field_path: str, message: str) -> FedotValidationError:
    return FedotValidationError(message, field_name=field_path)


def unknown_keys_error(prefix: str, keys: Set[str]) -> FedotInvalidKeysError:
    return FedotInvalidKeysError(keys, prefix=prefix)


def from_marshmallow_error(
    exc: ValidationError,
    *,
    prefix: str = '',
) -> FedotValidationError:
    if isinstance(exc, FedotValidationError):
        return exc

    unknown_keys = _extract_unknown_keys(exc)
    if unknown_keys is not None:
        if prefix:
            return unknown_keys_error(prefix, set(unknown_keys))
        return FedotInvalidKeysError(
            invalid_keys=set(unknown_keys),
            message=f'Invalid key parameters {unknown_keys}',
        )

    field_name = exc.field_name or '_schema'
    if prefix and field_name != '_schema':
        field_name = f'{prefix}.{field_name}'
    return FedotValidationError(exc.messages, field_name=field_name)


def _extract_unknown_keys(exc: ValidationError) -> Optional[FrozenSet[str]]:
    messages = exc.messages
    if not isinstance(messages, dict):
        return None

    unknown = set()
    for key, value in messages.items():
        if key == '_schema':
            continue
        if isinstance(value, list) and any('Unknown field' in str(item) for item in value):
            unknown.add(key)
        elif isinstance(value, dict):
            nested = _extract_unknown_keys(ValidationError(value, field_name=key))
            if nested:
                unknown.update(nested)
    return frozenset(unknown) if unknown else None
