"""One context-local registry; scopes publish a complete validated batch."""
from copy import deepcopy
from hashlib import sha256
import importlib
import inspect
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Iterable, Tuple

from pymonad.either import Left, Right
from pymonad.maybe import Just, Nothing

from fedot.extensions.call_rules import invoke_factory
from fedot.extensions.contracts import (
    ExtensionContractError, ExtensionError, ExtensionManifest, RegisteredExtension,
    unwrap_extension_result,
)
from fedot.extensions.registration_rules import plan_registration
from fedot.extensions.validation import (
    validate_extension_manifest, validate_external_model_spec, validate_external_transform_spec,
)

_REGISTERED_EXTENSIONS = ContextVar('fedot_extensions', default=())


def _factory_identity(factory) -> tuple[str, str, str]:
    target = inspect.unwrap(factory)
    module = getattr(target, '__module__', type(target).__module__)
    qualname = getattr(target, '__qualname__', type(target).__qualname__)
    try:
        implementation = inspect.getsource(target).encode('utf-8')
    except (OSError, TypeError):
        code = getattr(target, '__code__', None)
        implementation = code.co_code if code is not None else b''
    return module, qualname, sha256(implementation).hexdigest()


def registered_extensions_identity() -> str:
    """Return a stable identity for the active extension implementations."""
    from fedot.core.caching.normalization import stable_hash

    manifests = []
    for manifest in _REGISTERED_EXTENSIONS.get():
        manifests.append({
            'name': manifest.name,
            'version': manifest.version,
            'module': manifest.module,
            'models': tuple({
                'name': spec.name,
                'factory': _factory_identity(spec.factory),
                'capabilities': spec.capabilities,
                'hyperparams_schema': spec.hyperparams_schema,
            } for spec in manifest.models),
            'transforms': tuple({
                'name': spec.name,
                'factory': _factory_identity(spec.factory),
                'capabilities': spec.capabilities,
                'hyperparams_schema': spec.hyperparams_schema,
            } for spec in manifest.transforms),
        })
    return stable_hash(tuple(manifests), digest_size=32)


def _snapshot_manifest(manifest: ExtensionManifest) -> ExtensionManifest:
    """Copy mutable contract data while retaining live factory identities."""
    def snapshot_spec(spec):
        schema = replace(
            spec.hyperparams_schema,
            defaults=deepcopy(spec.hyperparams_schema.defaults),
        )
        return replace(spec, hyperparams_schema=schema)

    return replace(
        manifest,
        models=tuple(snapshot_spec(spec) for spec in manifest.models),
        transforms=tuple(snapshot_spec(spec) for spec in manifest.transforms),
    )


def _registered_view(manifest: ExtensionManifest) -> RegisteredExtension:
    return RegisteredExtension(_snapshot_manifest(manifest))


def _reserved_operation_names():
    # Read the active catalog without caching scoped extensions in it.
    from fedot.core.repository.operation_types_repository import OperationTypesRepository

    return tuple(operation.id for operation in OperationTypesRepository('all').operations)


def register_extensions(manifests: Iterable[ExtensionManifest], *, dry_run=False):
    manifests = tuple(manifests)
    current = _REGISTERED_EXTENSIONS.get()
    plan = plan_registration(manifests, current, _reserved_operation_names())
    if plan.is_left() or dry_run:
        return plan
    snapshots = tuple(_snapshot_manifest(manifest) for manifest in manifests)
    _REGISTERED_EXTENSIONS.set(current + snapshots)
    return Right(tuple(_registered_view(manifest) for manifest in snapshots))


def register_extension(manifest: ExtensionManifest):
    result = register_extensions((manifest,))
    return result if result.is_left() else Right(result.value[0])


@contextmanager
def extension_scope(*manifests: ExtensionManifest):
    """Inherit visible registrations, reject shadowing, restore on every exit."""
    token = _REGISTERED_EXTENSIONS.set(_REGISTERED_EXTENSIONS.get())
    try:
        registered = unwrap_extension_result(register_extensions(manifests))
        yield registered
    finally:
        _REGISTERED_EXTENSIONS.reset(token)


def get_registered_extensions() -> Tuple[RegisteredExtension, ...]:
    return tuple(_registered_view(manifest) for manifest in _REGISTERED_EXTENSIONS.get())


def get_registered_extension(extension_name: str):
    for manifest in _REGISTERED_EXTENSIONS.get():
        if manifest.name == extension_name:
            return Just(_registered_view(manifest))
    return Nothing


def clear_extension_registry() -> None:
    _REGISTERED_EXTENSIONS.set(())


def load_extension_manifest(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return Left(ExtensionError('module_import_failed',
                                   f'Unable to import extension module "{module_name}".', cause=exc))
    manifest = getattr(module, 'FEDOT_EXTENSION_MANIFEST', None)
    if manifest is None:
        return Left(ExtensionError('manifest_not_found',
                                   f'Module "{module_name}" must expose FEDOT_EXTENSION_MANIFEST.'))
    validation = validate_extension_manifest(manifest)
    if validation.is_left():
        return validation
    return Right(replace(manifest, module=module_name) if manifest.module is None else manifest)


def discover_extensions(module_names: Iterable[str]):
    manifests = []
    for module_name in module_names:
        loaded = load_extension_manifest(module_name)
        if loaded.is_left():
            return loaded
        manifests.append(loaded.value)
    return Right(tuple(manifests))


def smoke_test_extension(manifest: ExtensionManifest, parameters=None):
    """Factory smoke only; TensorData pipeline coverage lives in integration tests."""
    from fedot.extensions.parameter_rules import resolve_extension_params

    validation = validate_extension_manifest(manifest)
    if validation.is_left():
        return validation
    prepared = []
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
        return Left(ExtensionError('invalid_parameters', 'Smoke parameters must be an operation-keyed mapping.'))
    operation_names = {spec.name for spec in manifest.models + manifest.transforms}
    if set(parameters) - operation_names:
        return Left(ExtensionError('invalid_parameters', 'Smoke parameters contain an unknown operation.'))
    for spec in manifest.models + manifest.transforms:
        params = resolve_extension_params(spec, parameters.get(spec.name))
        if params.is_left():
            return params
        prepared.append((spec, params.value))
    for spec, params in prepared:
        try:
            instance = invoke_factory(spec.factory, params, 'factory_smoke_test_failed')
        except ExtensionContractError as exc:
            return Left(exc.error)
        if instance is None:
            return Left(ExtensionError('factory_returned_none',
                                       f'Factory for "{spec.name}" returned None.'))
    return Right(manifest)
