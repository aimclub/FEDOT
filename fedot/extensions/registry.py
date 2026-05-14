import importlib
import inspect
from typing import Any, Dict, Iterable, Tuple

from pymonad.either import Left, Right
from pymonad.maybe import Just, Nothing

from fedot.extensions.contracts import (
    ExtensionError,
    ExtensionManifest,
    ExternalModelSpec,
    RegisteredExtension,
)

_REGISTERED_EXTENSIONS: Dict[str, ExtensionManifest] = {}


def validate_extension_manifest(manifest: Any):
    if not isinstance(manifest, ExtensionManifest):
        return Left(ExtensionError(code='invalid_manifest_type',
                                   message='Extension manifest must be an ExtensionManifest instance.'))

    if not manifest.name.strip():
        return Left(ExtensionError(code='empty_extension_name',
                                   message='Extension manifest name must be non-empty.'))

    if not manifest.version.strip():
        return Left(ExtensionError(code='empty_extension_version',
                                   message='Extension manifest version must be non-empty.'))

    if not manifest.models:
        return Left(ExtensionError(code='empty_models',
                                   message='Extension manifest must expose at least one model.'))

    seen_names = set()
    for model in manifest.models:
        model_validation = validate_external_model_spec(model)
        if model_validation.is_left():
            return model_validation
        if model.name in seen_names:
            return Left(ExtensionError(code='duplicate_model_name',
                                       message=f'Duplicate model name "{model.name}" in extension manifest.',
                                       details={'extension': manifest.name}))
        seen_names.add(model.name)

    return Right(manifest)


def validate_external_model_spec(model: Any):
    if not isinstance(model, ExternalModelSpec):
        return Left(ExtensionError(code='invalid_model_spec_type',
                                   message='External model spec must be an ExternalModelSpec instance.'))

    if not model.name.strip():
        return Left(ExtensionError(code='empty_model_name',
                                   message='External model name must be non-empty.'))

    if not callable(model.factory):
        return Left(ExtensionError(code='invalid_model_factory',
                                   message=f'Factory for model "{model.name}" must be callable.'))

    if not model.capabilities.tasks:
        return Left(ExtensionError(code='empty_model_tasks',
                                   message=f'Model "{model.name}" must declare supported tasks.'))

    if not model.capabilities.data_types:
        return Left(ExtensionError(code='empty_model_data_types',
                                   message=f'Model "{model.name}" must declare supported data types.'))

    return Right(model)


def register_extension(manifest: ExtensionManifest):
    validation = validate_extension_manifest(manifest)
    if validation.is_left():
        return validation

    if manifest.name in _REGISTERED_EXTENSIONS:
        return Left(ExtensionError(code='duplicate_extension',
                                   message=f'Extension "{manifest.name}" is already registered.'))

    _REGISTERED_EXTENSIONS[manifest.name] = manifest
    return Right(RegisteredExtension(manifest=manifest))


def get_registered_extensions() -> Tuple[RegisteredExtension, ...]:
    return tuple(RegisteredExtension(manifest=manifest) for manifest in _REGISTERED_EXTENSIONS.values())


def get_registered_extension(extension_name: str):
    manifest = _REGISTERED_EXTENSIONS.get(extension_name)
    if manifest is None:
        return Nothing
    return Just(RegisteredExtension(manifest=manifest))


def clear_extension_registry() -> None:
    _REGISTERED_EXTENSIONS.clear()


def load_extension_manifest(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as ex:
        return Left(ExtensionError(code='module_import_failed',
                                   message=f'Unable to import extension module "{module_name}".',
                                   details={'exception': str(ex)}))

    manifest = getattr(module, 'FEDOT_EXTENSION_MANIFEST', None)
    if manifest is None:
        return Left(ExtensionError(code='manifest_not_found',
                                   message=f'Extension module "{module_name}" must expose FEDOT_EXTENSION_MANIFEST.'))

    if manifest.module is None:
        manifest = ExtensionManifest(name=manifest.name,
                                     version=manifest.version,
                                     models=manifest.models,
                                     module=module_name,
                                     description=manifest.description)
    return validate_extension_manifest(manifest)


def discover_extensions(module_names: Iterable[str]):
    manifests = []
    for module_name in module_names:
        loaded = load_extension_manifest(module_name)
        if loaded.is_left():
            return loaded
        manifests.append(loaded.value)
    return Right(tuple(manifests))


def smoke_test_extension(manifest: ExtensionManifest):
    validation = validate_extension_manifest(manifest)
    if validation.is_left():
        return validation

    for model in manifest.models:
        signature = inspect.signature(model.factory)
        positional_required = [
            parameter for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_required) > 1:
            return Left(ExtensionError(
                code='invalid_factory_signature',
                message=f'Factory for model "{model.name}" must accept zero or one positional argument.',
                details={'required_args': [parameter.name for parameter in positional_required]},
            ))

        try:
            instance = model.factory(None)
        except TypeError:
            instance = model.factory()
        except Exception as ex:
            return Left(ExtensionError(code='factory_smoke_test_failed',
                                       message=f'Factory smoke test failed for model "{model.name}".',
                                       details={'exception': str(ex)}))

        if instance is None:
            return Left(ExtensionError(code='factory_returned_none',
                                       message=f'Factory for model "{model.name}" returned None.'))

    return Right(manifest)

