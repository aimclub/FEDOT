import pytest
from marshmallow import ValidationError

from fedot.extensions.contracts import ModelHyperparamsSchema
from fedot.validation.boundaries import load_validated
from fedot.validation.errors import FedotInvalidKeysError, FedotValidationError
from fedot.extensions.schemas import (
    ExtensionManifestFieldsSchema,
    build_hyperparams_schema,
    validate_extension_hyperparams,
)


def test_validate_extension_hyperparams_with_defaults():
    """Required + optional params with defaults must all be validated together.

    Desired behavior: the user provides only the required 'lr'. The optional
    'epochs' has a default of 10 declared in the contract, so
    ``build_hyperparams_schema`` must synthesize a field with that default.
    The returned dict must contain both the user value and the filled-in default.
    """
    schema = ModelHyperparamsSchema(
        required=('lr',),
        optional=('epochs',),
        defaults={'epochs': 10},
    )
    result = validate_extension_hyperparams(schema, {'lr': 0.01})
    assert result['lr'] == 0.01
    assert result['epochs'] == 10


def test_validate_extension_hyperparams_missing_required_raises():
    """A missing required parameter must raise, not be recovered to a default.

    Desired behavior: 'lr' is required and has no default in the contract.
    The dynamically built schema must mark it ``required=True`` with no
    ``load_default``, so recovery cannot fill it in. The result is a
    ``FedotValidationError``.
    """
    schema = ModelHyperparamsSchema(required=('lr',), defaults={})
    with pytest.raises(FedotValidationError):
        validate_extension_hyperparams(schema, {})


def test_validate_extension_hyperparams_unknown_key_raises():
    """Keys not declared in the contract must be rejected as unknown keys.

    Desired behavior: the contract defines only 'lr', but the user passes
    'unknown'. Since the dynamically built schema uses RAISE mode, this must
    surface as ``FedotInvalidKeysError`` — catching typos in hyperparameter
    names that would otherwise be silently ignored.
    """
    schema = ModelHyperparamsSchema(required=('lr',), defaults={})
    with pytest.raises(FedotInvalidKeysError):
        validate_extension_hyperparams(schema, {'lr': 0.01, 'unknown': 1})


def test_build_hyperparams_schema_creates_schema_class():
    """The factory must produce a valid marshmallow Schema class.

    Desired behavior: ``build_hyperparams_schema`` dynamically constructs a
    schema class whose fields correspond to the union of required, optional,
    and default keys from the contract. The generated class must be
    instantiable and its fields dict must contain every declared key.
    """
    schema_cls = build_hyperparams_schema(
        ModelHyperparamsSchema(required=('lr',), defaults={'epochs': 5}))
    assert 'lr' in schema_cls().fields
    assert 'epochs' in schema_cls().fields


def test_extension_manifest_fields_schema_valid():
    """A manifest with valid name and version must load successfully.

    Desired behavior: the schema requires 'name' and 'version' and defaults
    description to '' and module to None. A dict with just the two required
    fields must pass, and the loaded dict must preserve them.
    """
    loaded = load_validated(
        ExtensionManifestFieldsSchema(),
        {'name': 'ext', 'version': '1.0.0'},
    )
    assert loaded['name'] == 'ext'
    assert loaded['version'] == '1.0.0'


def test_extension_manifest_fields_schema_empty_name_raises():
    """Whitespace-only 'name' must be rejected by the schema-level validator.

    Desired behavior: ``validates_schema`` checks that 'name' is not blank
    after stripping. '  ' (spaces only) must fail with
    ``FedotValidationError`` — the boundary converts the marshmallow error
    into the FEDOT error type.
    """
    with pytest.raises(FedotValidationError):
        load_validated(
            ExtensionManifestFieldsSchema(),
            {'name': '  ', 'version': '1.0.0'},
        )


def test_extension_manifest_fields_schema_empty_version_raises():
    """Empty string 'version' must be rejected.

    Desired behavior: this test calls ``schema.load()`` directly (bypassing the
    boundary's error conversion), so the raw ``ValidationError`` is expected.
    The empty-string '' fails the ``validates_schema`` check on 'version'.
    """
    schema = ExtensionManifestFieldsSchema()
    with pytest.raises(ValidationError, match='version'):
        schema.load({'name': 'ext', 'version': ''})
