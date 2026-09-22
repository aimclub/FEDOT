from contextvars import copy_context
from dataclasses import replace
from itertools import permutations
import sys
import types

import pytest

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ExtensionContractError, ExtensionManifest, ExternalModelSpec, ExternalTransformSpec,
    ModelCapabilities, TransformCapabilities, extension_scope, get_registered_extensions,
    load_extension_manifest, register_extension, register_extensions, validate_extension_manifest,
)
from fedot.extensions.registration_rules import plan_registration
from fedot.extensions.runtime_rules import get_extension_operation_spec


def manifest(name='test_extension', operation='test_model', factory=lambda: object()):
    return ExtensionManifest(name, '1', (ExternalModelSpec(
        operation, factory, ModelCapabilities((TaskTypesEnum.regression,), (DataTypesEnum.table,))),))


def transform_manifest(name='transform_extension', operation='test_transform'):
    return ExtensionManifest(name, '1', transforms=(ExternalTransformSpec(
        operation, lambda: object(), TransformCapabilities(
            (TaskTypesEnum.regression,), (DataTypesEnum.table,), DataTypesEnum.table)),))


def names():
    return {registered.manifest.name for registered in get_registered_extensions()}


def test_scope_restores_parent_on_success_and_exception():
    before = names()
    with extension_scope(manifest()):
        assert names() == before | {'test_extension'}
        with pytest.raises(RuntimeError, match='body failed'):
            with extension_scope(transform_manifest()):
                assert names() == before | {'test_extension', 'transform_extension'}
                raise RuntimeError('body failed')
        assert names() == before | {'test_extension'}
    assert names() == before


def test_failed_batch_publishes_nothing_and_does_not_shadow():
    with extension_scope(manifest()):
        before = get_registered_extensions()
        with pytest.raises(ExtensionContractError) as error:
            with extension_scope(transform_manifest(), manifest('other')):
                pytest.fail('conflict must be detected before entering scope')
        assert error.value.code == 'operation_name_conflict'
        assert get_registered_extensions() == before
        assert get_extension_operation_spec('test_transform') is None


@pytest.mark.parametrize('operation', ['linear', 'scaling'])
def test_builtin_name_conflicts_are_rejected(operation):
    before = get_registered_extensions()
    result = register_extension(manifest(operation=operation))
    assert result.is_left()
    assert result.monoid[0].code == 'operation_name_conflict'
    assert get_registered_extensions() == before


def test_model_and_transform_share_one_name_namespace():
    with extension_scope(manifest()):
        result = register_extension(transform_manifest(operation='test_model'))
        assert result.monoid[0].code == 'operation_name_conflict'


def test_duplicate_within_manifest_and_across_batch():
    model = manifest()
    transform = transform_manifest(operation='test_model')
    mixed = replace(model, transforms=transform.transforms)
    assert validate_extension_manifest(mixed).monoid[0].code == 'duplicate_model_name'
    assert plan_registration((model, transform)).monoid[0].code == 'operation_name_conflict'


def test_registration_order_does_not_change_lookup():
    manifests = (manifest(), transform_manifest(), manifest('second', 'second_model'))
    expected = {spec.name for item in manifests for spec in item.models + item.transforms}
    for order in permutations(manifests):
        with extension_scope(*order):
            assert {get_extension_operation_spec(name).name for name in expected} == expected


def test_duplicate_conflict_is_monotonic():
    first = manifest()
    conflict = manifest('other')
    for suffix in ((), (transform_manifest(),), (manifest('third', 'third_model'),)):
        result = plan_registration((first, conflict) + suffix)
        assert result.monoid[0].code == 'operation_name_conflict'


def test_context_copy_cannot_mutate_parent_registry():
    with extension_scope(manifest()):
        child = copy_context()
        assert child.run(register_extension, transform_manifest()).is_right()
        assert get_extension_operation_spec('test_transform') is None
        assert child.run(get_extension_operation_spec, 'test_transform') is not None


def test_dry_run_produces_only_descriptors_without_factory_effects():
    calls = []
    item = manifest(factory=lambda: calls.append('factory'))
    before = get_registered_extensions()
    plan = register_extensions((item,), dry_run=True)
    assert plan.is_right()
    assert plan.value.extension_names == ('test_extension',)
    assert plan.value.entries[0].operation_name == 'test_model'
    assert calls == []
    assert get_registered_extensions() == before


@pytest.mark.parametrize('payload,code', [(42, 'invalid_manifest_type'),
                                          (None, 'manifest_not_found')])
def test_discovery_rejects_invalid_payload_before_accessing_attributes(monkeypatch, payload, code):
    module = types.ModuleType('invalid_manifest_module')
    module.FEDOT_EXTENSION_MANIFEST = payload
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert load_extension_manifest(module.__name__).monoid[0].code == code


def test_discovery_retains_transform_specs(monkeypatch):
    module = types.ModuleType('transform_manifest_module')
    item = transform_manifest()
    module.FEDOT_EXTENSION_MANIFEST = item
    monkeypatch.setitem(sys.modules, module.__name__, module)
    loaded = load_extension_manifest(module.__name__).value
    assert loaded.transforms == item.transforms
    assert loaded.module == module.__name__
