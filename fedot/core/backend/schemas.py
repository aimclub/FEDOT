import re
from typing import Any

from marshmallow import RAISE, Schema, ValidationError, fields, validates

from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext

BACKEND_CUDA_DEVICE_PATTERN = re.compile(r'^cuda:\d+$')
BACKEND_SUPPORTED_NAMES_HINT = 'cpu, gpu, cuda, mps, cuda:<device_index>'


class BackendNameSchema(Schema):
    class Meta:
        unknown = RAISE

    name = fields.Raw(required=True)

    @validates('name')
    def validate_name(self, value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(
                f'Backend name must be a non-empty string, got {value!r}. '
                f'Expected one of: {BACKEND_SUPPORTED_NAMES_HINT}'
            )

        normalized = value.strip().lower()
        if normalized in ('cpu', 'gpu', 'cuda', 'mps'):
            return
        if BACKEND_CUDA_DEVICE_PATTERN.match(normalized):
            return

        raise ValidationError(
            f'Unsupported backend name: {value!r}. '
            f'Expected one of: {BACKEND_SUPPORTED_NAMES_HINT}'
        )


def validate_backend_name(name: Any, context: ValidationContext = None) -> str:
    validated = load_validated(
        BackendNameSchema(),
        {'name': name},
        context,
        prefix='backend',
    )
    return validated['name'].strip().lower()
