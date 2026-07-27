from typing import Any, Dict, Type

from marshmallow import RAISE, Schema, ValidationError, fields, validates_schema

from fedot.extensions.contracts import ModelHyperparamsSchema
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext


def build_hyperparams_schema(hyperparams_schema: ModelHyperparamsSchema) -> Type[Schema]:
    allowed_keys = (
        set(hyperparams_schema.required)
        | set(hyperparams_schema.optional)
        | set(hyperparams_schema.defaults.keys())
    )
    attrs = {
        'Meta': type('Meta', (), {'unknown': RAISE}),
    }
    for key in allowed_keys:
        default = hyperparams_schema.defaults.get(key)
        is_required = key in hyperparams_schema.required and default is None
        if default is not None:
            attrs[key] = fields.Raw(load_default=default, allow_none=True)
        elif is_required:
            attrs[key] = fields.Raw(required=True)
        else:
            attrs[key] = fields.Raw(allow_none=True)
    return type('ExtensionHyperparamsSchema', (Schema,), attrs)


def validate_extension_hyperparams(
    hyperparams_schema: ModelHyperparamsSchema,
    params: Dict[str, Any],
    context: ValidationContext = None,
) -> Dict[str, Any]:
    schema_cls = build_hyperparams_schema(hyperparams_schema)
    return load_validated(schema_cls(), params, context, prefix='extension_hyperparams')


class ExtensionManifestFieldsSchema(Schema):
    class Meta:
        unknown = RAISE

    name = fields.Str(required=True)
    version = fields.Str(required=True)
    description = fields.Str(load_default='')
    module = fields.Str(allow_none=True, load_default=None)

    @validates_schema
    def validate_non_empty(self, data: Dict[str, Any], **kwargs) -> None:
        if not str(data.get('name', '')).strip():
            raise ValidationError('Extension manifest name must be non-empty.', field_name='name')
        if not str(data.get('version', '')).strip():
            raise ValidationError('Extension manifest version must be non-empty.', field_name='version')
