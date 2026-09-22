from dataclasses import replace
from itertools import permutations
import sys
import types

import pytest

from fedot.core.operations.extension_model import ExtensionModel
from fedot.core.operations.extension_transform import ExtensionTransform
from fedot.core.operations.factory import OperationFactory
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_query import RepositoryKind
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions.call_rules import invoke_factory, invoke_once
from fedot.extensions.contracts import (
    ExtensionContractError,
    ExtensionManifest,
    ExternalModelSpec,
    ExternalTransformSpec,
    ModelCapabilities,
    OperationKind,
    TransformCapabilities,
)
from fedot.extensions.operation_rules import get_extension_operation_names
from fedot.extensions.registry import (
    discover_extensions,
    extension_scope,
    get_registered_extensions,
    register_extension,
    register_extensions,
)
from fedot.extensions.runtime_rules import (
    get_extension_model_spec,
    get_extension_operation_spec,
    get_extension_transform_spec,
)


def _default_factory():
    return object()


def _manifest(extension_name, operation_name, kind=OperationKind.model, factory=_default_factory):
    if kind is OperationKind.model:
        models = (ExternalModelSpec(
            operation_name,
            factory,
            ModelCapabilities((TaskTypesEnum.regression,), (DataTypesEnum.table,)),
        ),)
        transforms = ()
    else:
        models = ()
        transforms = (ExternalTransformSpec(
            operation_name,
            factory,
            TransformCapabilities(
                (TaskTypesEnum.regression,),
                (DataTypesEnum.table,),
                DataTypesEnum.table,
            ),
        ),)
    return ExtensionManifest(extension_name, '1', models=models, transforms=transforms)


def _package(size, prefix='fed02_batch', factory=_default_factory):
    return tuple(
        _manifest(
            f'{prefix}_extension_{index}',
            f'{prefix}_operation_{index}',
            OperationKind.model if index % 2 == 0 else OperationKind.transform,
            factory,
        )
        for index in range(size)
    )


def _rename_operation(manifest, operation_name):
    if manifest.models:
        return replace(manifest, models=(replace(manifest.models[0], name=operation_name),))
    return replace(manifest, transforms=(replace(manifest.transforms[0], name=operation_name),))


_CONFLICT_CASES = tuple(
    (size, conflict_index, conflict_kind)
    for size in range(1, 6)
    for conflict_index in range(size)
    for conflict_kind in ('extension', 'operation')
)


@pytest.mark.parametrize('size,conflict_index,conflict_kind', _CONFLICT_CASES)
def test_registration_batch_is_atomic_for_conflict_at_any_position(
        size, conflict_index, conflict_kind):
    parent = _manifest('fed02_parent_extension', 'fed02_parent_operation')
    package = list(_package(size))
    if conflict_kind == 'extension':
        package[conflict_index] = replace(package[conflict_index], name=parent.name)
        expected_code = 'duplicate_extension'
    else:
        package[conflict_index] = _rename_operation(
            package[conflict_index], parent.models[0].name)
        expected_code = 'operation_name_conflict'

    with extension_scope(parent):
        before = get_registered_extensions()
        result = register_extensions(package)

        assert result.is_left()
        assert result.monoid[0].code == expected_code
        assert get_registered_extensions() == before
        assert get_extension_operation_spec(parent.models[0].name) == parent.models[0]
        for index in range(size):
            assert get_extension_operation_spec(f'fed02_batch_operation_{index}') is None


@pytest.mark.parametrize('size', range(1, 7))
def test_dry_run_is_deterministic_order_independent_and_effect_free(size):
    calls = []

    def factory():
        calls.append('factory')
        return object()

    package = _package(size, prefix='fed02_plan', factory=factory)
    before = get_registered_extensions()

    forward = register_extensions(package, dry_run=True)
    repeated = register_extensions(package, dry_run=True)
    reverse = register_extensions(reversed(package), dry_run=True)

    assert forward.is_right() and repeated.is_right() and reverse.is_right()
    assert forward.value == repeated.value
    assert {
        entry.operation_name: entry.kind for entry in forward.value.entries
    } == {
        entry.operation_name: entry.kind for entry in reverse.value.entries
    }
    assert set(forward.value.extension_names) == set(reverse.value.extension_names)
    assert calls == []
    assert get_registered_extensions() == before


def test_nested_scope_restores_parent_after_direct_registration_and_exception():
    parent = _manifest('fed02_scope_parent', 'fed02_scope_parent_model')
    outer_direct = _manifest('fed02_scope_outer', 'fed02_scope_outer_transform', OperationKind.transform)
    inner = _manifest('fed02_scope_inner', 'fed02_scope_inner_model')
    inner_direct = _manifest('fed02_scope_direct', 'fed02_scope_direct_transform', OperationKind.transform)
    initial = get_registered_extensions()

    with extension_scope(parent):
        assert register_extension(outer_direct).is_right()
        outer_state = get_registered_extensions()

        with pytest.raises(RuntimeError, match='leave inner scope'):
            with extension_scope(inner):
                assert register_extension(inner_direct).is_right()
                raise RuntimeError('leave inner scope')

        assert get_registered_extensions() == outer_state
        assert get_extension_operation_spec(parent.models[0].name) == parent.models[0]
        assert get_extension_operation_spec(outer_direct.transforms[0].name) == outer_direct.transforms[0]
        assert get_extension_operation_spec(inner.models[0].name) is None
        assert get_extension_operation_spec(inner_direct.transforms[0].name) is None

    assert get_registered_extensions() == initial


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError])
@pytest.mark.parametrize('invocation', ['factory', 'method'])
def test_non_type_error_from_user_code_is_not_retried(error_type, invocation):
    calls = []
    cause = error_type('user code failed')

    def user_code(*args, **kwargs):
        calls.append((args, kwargs))
        raise cause

    with pytest.raises(ExtensionContractError) as error:
        if invocation == 'factory':
            invoke_factory(user_code, {'value': 1})
        else:
            invoke_once(user_code, (((1,), {}), ((1, 2), {}), ((), {})))

    assert error.value.code == 'extension_execution_failed'
    assert error.value.__cause__ is cause
    assert error.value.error.cause is cause
    assert len(calls) == 1


def test_discovery_and_operation_factory_preserve_operation_kinds_in_any_module_order(monkeypatch):
    calls = []

    def factory():
        calls.append('factory')
        return object()

    model_name = 'fed02_discovered_model'
    transform_name = 'fed02_discovered_transform'
    modules = (
        ('fed02_model_module', _manifest('fed02_model_extension', model_name, factory=factory)),
        ('fed02_transform_module', _manifest(
            'fed02_transform_extension', transform_name, OperationKind.transform, factory)),
    )
    for module_name, manifest in modules:
        module = types.ModuleType(module_name)
        module.FEDOT_EXTENSION_MANIFEST = manifest
        monkeypatch.setitem(sys.modules, module_name, module)

    for module_order in permutations(module_name for module_name, _ in modules):
        discovered = discover_extensions(module_order)
        assert discovered.is_right()

        with extension_scope(*discovered.value):
            assert isinstance(get_extension_model_spec(model_name), ExternalModelSpec)
            assert get_extension_transform_spec(model_name) is None
            assert isinstance(get_extension_transform_spec(transform_name), ExternalTransformSpec)
            assert get_extension_model_spec(transform_name) is None
            assert isinstance(OperationFactory(f'{model_name}/node').get_operation(), ExtensionModel)
            assert isinstance(OperationFactory(f'{transform_name}/node').get_operation(), ExtensionTransform)

            model_names = get_extension_operation_names(
                TaskTypesEnum.regression,
                DataTypesEnum.table,
                repository_kind=RepositoryKind.model,
            )
            transform_names = get_extension_operation_names(
                TaskTypesEnum.regression,
                DataTypesEnum.table,
                repository_kind=RepositoryKind.data_operation,
            )
            assert model_name in model_names and transform_name not in model_names
            assert transform_name in transform_names and model_name not in transform_names

    assert calls == []
