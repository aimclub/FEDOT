"""Manifest rules shared by registration, discovery and runtime preparation."""
from collections.abc import Mapping

from pymonad.either import Left, Right

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.call_rules import FACTORY_SHAPES, inspect_call
from fedot.extensions.contracts import (
    ArrayBackend, ExtensionError, ExtensionManifest, ExternalModelSpec,
    ExternalTransformSpec, ModelCapabilities, ModelHyperparamsSchema, TransformCapabilities,
)

RESERVED_BEHAVIOR_TAGS = frozenset({'correct_params'})


def _failure(code, message, **details):
    return Left(ExtensionError(code, message, details))


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def validate_hyperparams_schema(schema):
    if not isinstance(schema, ModelHyperparamsSchema):
        return _failure('invalid_hyperparams_schema', 'Expected ModelHyperparamsSchema.')
    if not isinstance(schema.required, tuple) or not isinstance(schema.optional, tuple):
        return _failure('invalid_hyperparams_schema', 'Parameter names must be tuples.')
    if not isinstance(schema.defaults, Mapping):
        return _failure('invalid_hyperparams_schema', 'Parameter defaults must be a mapping.')
    names = schema.required + schema.optional
    for name in names + tuple(schema.defaults):
        if not _name(name) or name.startswith('_') or name in ('model_fit', 'model_predict'):
            return _failure('invalid_hyperparams_schema', 'Invalid or reserved parameter name.')
    if len(names) != len(set(names)):
        return _failure('invalid_hyperparams_schema', 'Parameter declarations must not overlap.')
    return Right(schema)


def _validate_spec(spec, spec_type, capabilities_type, kind):
    if not isinstance(spec, spec_type):
        return _failure(f'invalid_{kind}_spec_type', f'Expected {spec_type.__name__}.')
    if not _name(spec.name):
        return _failure(f'empty_{kind}_name', 'Operation name must be non-empty.')
    if spec.name != spec.name.strip() or '/' in spec.name:
        return _failure('invalid_operation_name', 'Operation name cannot contain a suffix or outer whitespace.')
    if not callable(spec.factory):
        return _failure(f'invalid_{kind}_factory', 'Operation factory must be callable.')
    caps = spec.capabilities
    if not isinstance(caps, capabilities_type):
        return _failure('invalid_capabilities', f'Expected {capabilities_type.__name__}.')
    for field, enum in (('tasks', TaskTypesEnum), ('data_types', DataTypesEnum)):
        values = getattr(caps, field)
        if not isinstance(values, tuple):
            return _failure('invalid_capabilities', f'{field} must be a tuple.')
        if not values:
            return _failure(f'empty_{kind}_{field}', f'Operation must declare {field}.')
        if not all(isinstance(value, enum) for value in values):
            return _failure('invalid_capabilities', f'Invalid {field} member.')
    if not isinstance(caps.tags, tuple) or not all(_name(tag) for tag in caps.tags):
        return _failure('invalid_capabilities', 'Tags must be non-empty strings.')
    reserved_tags = tuple(sorted(RESERVED_BEHAVIOR_TAGS.intersection(caps.tags)))
    if reserved_tags:
        return _failure(
            'reserved_behavior_tag',
            'Extension capability tags must not alter FEDOT runtime behavior.',
            operation=spec.name,
            tags=list(reserved_tags),
        )
    if not isinstance(caps.backend, ArrayBackend):
        return _failure('invalid_capabilities', 'Backend must be an ArrayBackend.')
    flags = (caps.supports_multimodal, caps.requires_target)
    if isinstance(caps, TransformCapabilities):
        flags += (caps.requires_fit,)
    if not all(isinstance(flag, bool) for flag in flags):
        return _failure('invalid_capabilities', 'Capability flags must be boolean.')
    if not isinstance(caps.output_data_type, DataTypesEnum):
        if not isinstance(caps, ModelCapabilities) or caps.output_data_type is not None:
            return _failure('invalid_capabilities', 'Output data type must be declared.')
    schema_result = validate_hyperparams_schema(spec.hyperparams_schema)
    if schema_result.is_left():
        return schema_result
    signature_result = inspect_call(spec.factory, FACTORY_SHAPES, 'invalid_factory_signature')
    return signature_result if signature_result.is_left() else Right(spec)


def validate_external_model_spec(model):
    return _validate_spec(model, ExternalModelSpec, ModelCapabilities, 'model')


def validate_external_transform_spec(transform):
    return _validate_spec(transform, ExternalTransformSpec, TransformCapabilities, 'transform')


def validate_operation_spec(spec):
    if isinstance(spec, ExternalTransformSpec):
        return validate_external_transform_spec(spec)
    return validate_external_model_spec(spec)


def validate_extension_manifest(manifest):
    if not isinstance(manifest, ExtensionManifest):
        return _failure('invalid_manifest_type', 'Expected ExtensionManifest.')
    if not _name(manifest.name):
        return _failure('empty_extension_name', 'Extension name must be non-empty.')
    if not _name(manifest.version):
        return _failure('empty_extension_version', 'Extension version must be non-empty.')
    if not isinstance(manifest.models, tuple) or not isinstance(manifest.transforms, tuple):
        return _failure('invalid_manifest_operations', 'Models and transforms must be tuples.')
    if not manifest.models and not manifest.transforms:
        return _failure('empty_models', 'Extension must expose a model or transform.')
    seen = set()
    for specs, validator in ((manifest.models, validate_external_model_spec),
                             (manifest.transforms, validate_external_transform_spec)):
        for spec in specs:
            result = validator(spec)
            if result.is_left():
                return result
            if spec.name in seen:
                return _failure('duplicate_model_name', 'Duplicate operation name in manifest.',
                                extension=manifest.name, operation=spec.name)
            seen.add(spec.name)
    return Right(manifest)
