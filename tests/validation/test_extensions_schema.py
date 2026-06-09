import pytest
from marshmallow import ValidationError

from fedot.extensions.contracts import ModelHyperparamsSchema
from fedot.validation.boundaries import load_validated
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.validation.schemas.extensions import (
    ExtensionManifestFieldsSchema,
    build_hyperparams_schema,
    validate_extension_hyperparams,
)


def test_validate_extension_hyperparams_with_defaults():
    schema = ModelHyperparamsSchema(
        required=('lr',),
        optional=('epochs',),
        defaults={'epochs': 10},
    )
    result = validate_extension_hyperparams(schema, {'lr': 0.01})
    assert result['lr'] == 0.01
    assert result['epochs'] == 10


def test_validate_extension_hyperparams_missing_required_raises():
    schema = ModelHyperparamsSchema(required=('lr',), defaults={})
    with pytest.raises(FedotValidationError):
        validate_extension_hyperparams(schema, {})


def test_validate_extension_hyperparams_unknown_key_raises():
    schema = ModelHyperparamsSchema(required=('lr',), defaults={})
    with pytest.raises(FedotInvalidKeysError):
        validate_extension_hyperparams(schema, {'lr': 0.01, 'unknown': 1})


def test_build_hyperparams_schema_creates_schema_class():
    schema_cls = build_hyperparams_schema(
        ModelHyperparamsSchema(required=('lr',), defaults={'epochs': 5}))
    assert 'lr' in schema_cls().fields
    assert 'epochs' in schema_cls().fields


def test_extension_manifest_fields_schema_valid():
    loaded = load_validated(
        ExtensionManifestFieldsSchema(),
        {'name': 'ext', 'version': '1.0.0'},
    )
    assert loaded['name'] == 'ext'
    assert loaded['version'] == '1.0.0'


def test_extension_manifest_fields_schema_empty_name_raises():
    with pytest.raises(FedotValidationError):
        load_validated(
            ExtensionManifestFieldsSchema(),
            {'name': '  ', 'version': '1.0.0'},
        )


def test_extension_manifest_fields_schema_empty_version_raises():
    schema = ExtensionManifestFieldsSchema()
    with pytest.raises(ValidationError, match='version'):
        schema.load({'name': 'ext', 'version': ''})
