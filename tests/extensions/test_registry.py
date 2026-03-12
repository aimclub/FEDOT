import sys
import types

from pymonad.either import Left, Right
from pymonad.maybe import Just, Nothing

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ExtensionManifest,
    ExternalModelSpec,
    ModelCapabilities,
    clear_extension_registry,
    discover_extensions,
    get_registered_extension,
    get_registered_extensions,
    register_extension,
    smoke_test_extension,
    validate_extension_manifest,
)


def _dummy_factory(_params=None):
    return object()


def _build_manifest(name='demo_extension', model_name='demo_model'):
    return ExtensionManifest(
        name=name,
        version='0.1.0',
        models=(
            ExternalModelSpec(
                name=model_name,
                factory=_dummy_factory,
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.classification,),
                    data_types=(DataTypesEnum.table,),
                    tags=('demo',),
                ),
                description='demo external model',
            ),
        ),
        description='demo extension',
    )


def setup_function():
    clear_extension_registry()


def teardown_function():
    clear_extension_registry()


def test_validate_extension_manifest_returns_right_for_valid_manifest():
    result = validate_extension_manifest(_build_manifest())

    assert result.__class__ is Right


def test_register_extension_stores_manifest_and_returns_maybe_lookup():
    manifest = _build_manifest()

    result = register_extension(manifest)

    assert result.__class__ is Right
    assert len(get_registered_extensions()) == 1
    assert get_registered_extension('demo_extension').__class__ is Just
    assert get_registered_extension('missing_extension').__class__ is Nothing


def test_register_extension_rejects_duplicate_extension_name():
    manifest = _build_manifest()
    register_extension(manifest)

    duplicate_result = register_extension(manifest)

    assert duplicate_result.__class__ is Left
    assert duplicate_result.value.code == 'duplicate_extension'


def test_smoke_test_extension_rejects_factory_returning_none():
    manifest = ExtensionManifest(
        name='broken_extension',
        version='0.1.0',
        models=(
            ExternalModelSpec(
                name='broken_model',
                factory=lambda _params=None: None,
                capabilities=ModelCapabilities(
                    tasks=(TaskTypesEnum.regression,),
                    data_types=(DataTypesEnum.table,),
                ),
            ),
        ),
    )

    result = smoke_test_extension(manifest)

    assert result.__class__ is Left
    assert result.value.code == 'factory_returned_none'


def test_discover_extensions_loads_manifest_from_module():
    module_name = 'tests.extensions.fake_extension_module'
    module = types.ModuleType(module_name)
    module.FEDOT_EXTENSION_MANIFEST = _build_manifest(name='module_extension')
    sys.modules[module_name] = module

    try:
        result = discover_extensions((module_name,))
        assert result.__class__ is Right
        assert result.value[0].module == module_name
    finally:
        del sys.modules[module_name]
