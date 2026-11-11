import json

import pytest
import torch

from fedot.core.operations.operation_execution_rules import ExecutionEnvironment, resolve_operation_execution
from fedot.core.repository import operation_implementation as implementation_module
from fedot.core.repository import operation_types_repository as repository_module
from fedot.core.repository.operation_implementation import (
    DeviceType,
    ExecutionPlatform,
    StrategyReference,
    parse_operation_implementations,
)
from fedot.core.repository.operation_types_repository import OperationTypesRepository, create_repository_path
from fedot.core.repository.tasks import TaskTypesEnum


@pytest.fixture(scope='module')
def model_operations():
    return {operation.id: operation for operation in OperationTypesRepository._initialise_repo(
        create_repository_path('model_repository.json'))}


@pytest.mark.parametrize('name,task,strategy', [
    ('rf', TaskTypesEnum.classification, 'Classification'),
    ('rfr', TaskTypesEnum.regression, 'Regression'),
])
def test_forests_keep_legacy_strategy_and_expose_lazy_variants(model_operations, name, task, strategy):
    operation = model_operations[name]
    cpu, gpu = operation.implementations

    assert operation.current_strategy(task).__name__ == f'Sklearn{strategy}Strategy'
    assert (cpu.id, cpu.devices, cpu.requires) == ('sklearn', (DeviceType.cpu,), ('sklearn',))
    assert cpu.strategy_for_task(task).name == f'Sklearn{strategy}Strategy'
    assert (gpu.id, gpu.devices, gpu.platforms) == ('cuml', (DeviceType.cuda,), (ExecutionPlatform.linux,))
    assert gpu.requires == ('cuml', 'cupy')
    assert gpu.strategy_for_task(task).name == f'CuML{strategy}Strategy'
    assert operation.presets == ['fast_train', '*tree']
    assert operation.tags[-2:] == ['tree', 'non_linear']


def test_unmigrated_operation_does_not_acquire_assumed_cpu_capabilities(model_operations):
    assert model_operations['ar'].implementations == ()
    assert model_operations['cnn'].implementations == ()
    assert model_operations['custom'].implementations == ()


@pytest.mark.parametrize('name,task', [('rf', TaskTypesEnum.classification), ('rfr', TaskTypesEnum.regression)])
@pytest.mark.parametrize('device,platform,expected', [
    ('cpu', ExecutionPlatform.linux, 'sklearn'),
    ('cuda:1', ExecutionPlatform.linux, 'cuml'),
    ('cuda:1', ExecutionPlatform.windows, 'sklearn'),
])
def test_real_catalog_drives_selection_without_instantiating_models(model_operations, name, task,
                                                                   device, platform, expected):
    environment = ExecutionEnvironment(platform, frozenset((DeviceType.cpu, DeviceType.cuda)),
                                       frozenset(('sklearn', 'cuml', 'cupy')))
    result = resolve_operation_execution(model_operations[name], task, torch.device(device), environment)
    assert result.implementation_id == expected


def test_catalog_loading_does_not_import_optional_variant_strategies(monkeypatch):
    imports = []

    def import_legacy_strategy(reference):
        imports.append(reference[0])
        assert reference[0] != 'fedot.core.operations.strategies.models.cuml'
        return object

    monkeypatch.setattr(repository_module, 'import_strategy_from_str', import_legacy_strategy)
    operations = OperationTypesRepository._initialise_repo(create_repository_path('model_repository.json'))

    assert imports
    assert next(operation for operation in operations if operation.id == 'rf').implementations[1].id == 'cuml'


def test_strategy_reference_imports_only_on_explicit_load(monkeypatch):
    imports = []
    strategy_class = object()

    def load(reference):
        imports.append(reference)
        return strategy_class

    monkeypatch.setattr(implementation_module, 'import_strategy_from_str', load)
    reference = StrategyReference('optional.module', 'Strategy')

    assert imports == []
    assert reference.load() is strategy_class
    assert imports == [['optional.module', 'Strategy']]


def test_shared_definitions_support_local_overrides_and_task_specific_references():
    shared = {'devices': ['cpu'], 'platforms': ['linux'], 'requires': ['dependency'],
              'strategies': ['unused.module', 'UnusedStrategy']}
    source = {'implementations': {'portable': {
        'meta': 'shared', 'devices': ['cpu', 'cuda'], 'platforms': [],
        'strategies': {'TaskTypesEnum.classification': ['fake.module', 'Classifier'],
                       'TaskTypesEnum.regression': ['fake.module', 'Regressor']},
    }}}
    before = json.dumps((source, shared), sort_keys=True)

    implementation, = parse_operation_implementations(source, {'shared': shared},
                                                       (TaskTypesEnum.classification, TaskTypesEnum.regression))

    assert implementation.devices == (DeviceType.cpu, DeviceType.cuda)
    assert implementation.platforms == ()
    assert implementation.requires == ('dependency',)
    assert implementation.strategy_for_task(TaskTypesEnum.classification).name == 'Classifier'
    assert implementation.strategy_for_task(TaskTypesEnum.regression).name == 'Regressor'
    assert implementation.strategy_for_task(TaskTypesEnum.clustering) is None
    assert json.dumps((source, shared), sort_keys=True) == before


@pytest.mark.parametrize('field,value', [('devices', ['gpu']), ('platforms', ['unknown-os'])])
def test_invalid_capabilities_fail_at_catalog_boundary(field, value):
    definition = {'devices': ['cpu'], 'strategies': ['fake.module', 'Strategy'], field: value}
    with pytest.raises(ValueError):
        parse_operation_implementations({'implementations': {'invalid': definition}}, {},
                                        (TaskTypesEnum.classification,))


def test_malformed_strategy_definition_fails_at_catalog_boundary():
    with pytest.raises(TypeError, match='Implementation strategies'):
        parse_operation_implementations({'implementations': {'invalid': {'devices': ['cpu'], 'strategies': 42}}},
                                        {}, (TaskTypesEnum.classification,))
