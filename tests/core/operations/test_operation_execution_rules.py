from dataclasses import replace

import pytest
import torch

from fedot.core.operations.operation_execution_rules import (
    ExecutionEnvironment,
    resolve_operation_execution,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_implementation import (
    DeviceType,
    ExecutionPlatform,
    OperationImplementationInfo,
    StrategyReference,
)
from fedot.core.repository.operation_types_repository import OperationMetaInfo
from fedot.core.repository.tasks import TaskTypesEnum


TASK = TaskTypesEnum.classification
CPU_STRATEGY = StrategyReference('not_imported.sklearn', 'Classifier')
GPU_STRATEGY = StrategyReference('not_imported.cuml', 'Classifier')
CPU = OperationImplementationInfo('sklearn', (DeviceType.cpu,), ((TASK, CPU_STRATEGY),), requires=('sklearn',))
GPU = OperationImplementationInfo('cuml', (DeviceType.cuda,), ((TASK, GPU_STRATEGY),),
                                  platforms=(ExecutionPlatform.linux,), requires=('cuml', 'cupy'))
ENVIRONMENT = ExecutionEnvironment(ExecutionPlatform.linux, frozenset((DeviceType.cpu, DeviceType.cuda)),
                                   frozenset(('sklearn', 'cuml', 'cupy')))


def operation(*implementations, tasks=(TASK,)):
    return OperationMetaInfo(id='example', input_types=[DataTypesEnum.tabular], output_types=[DataTypesEnum.tabular],
                             task_type=list(tasks), supported_strategies=object, allowed_positions=['any'],
                             implementations=implementations)


def resolve(metadata, device='cuda:1', environment=ENVIRONMENT, task=TASK):
    return resolve_operation_execution(metadata, task, torch.device(device), environment)


@pytest.mark.parametrize('platform,expected', [
    (ExecutionPlatform.linux, 'cuml'),
    (ExecutionPlatform.windows, 'sklearn'),
    (ExecutionPlatform.darwin, 'sklearn'),
    (ExecutionPlatform.wsl, 'sklearn'),
])
def test_gpu_selection_respects_platform_metadata(platform, expected):
    result = resolve(operation(CPU, GPU), environment=replace(ENVIRONMENT, platform=platform))

    assert result.implementation_id == expected
    assert result.device == torch.device('cuda:1' if expected == 'cuml' else 'cpu')


def test_cpu_input_does_not_enable_gpu_even_when_gpu_variant_is_first():
    result = resolve(operation(GPU, CPU), device='cpu')

    assert result.implementation_id == 'sklearn'
    assert result.device == torch.device('cpu')
    assert result.strategy == CPU_STRATEGY


@pytest.mark.parametrize('dependencies', [frozenset(('sklearn',)), frozenset(('sklearn', 'cuml')),
                                            frozenset(('sklearn', 'cupy'))])
def test_unavailable_gpu_dependencies_select_cpu_before_training(dependencies):
    result = resolve(operation(CPU, GPU), environment=replace(ENVIRONMENT, available_dependencies=dependencies))
    assert result.implementation_id == 'sklearn'


def test_unavailable_cuda_selects_cpu():
    result = resolve(operation(CPU, GPU), environment=replace(ENVIRONMENT,
                                                            available_devices=frozenset((DeviceType.cpu,))))
    assert result.implementation_id == 'sklearn'


def test_cpu_only_operation_remains_available_for_cuda_input():
    assert resolve(operation(CPU)).device == torch.device('cpu')


@pytest.mark.parametrize('device', ['cpu', 'cuda:2', 'mps'])
def test_portable_implementation_keeps_concrete_input_device(device):
    portable = replace(CPU, id='torch', devices=(DeviceType.cpu, DeviceType.cuda, DeviceType.mps), requires=())
    environment = replace(ENVIRONMENT, available_devices=frozenset(DeviceType))
    result = resolve(operation(portable), device=device, environment=environment)

    assert result.implementation_id == 'torch'
    assert result.device == torch.device(device)


def test_mps_does_not_select_cuda_variant():
    result = resolve(operation(CPU, GPU), device='mps',
                     environment=replace(ENVIRONMENT, available_devices=frozenset(DeviceType)))
    assert result.implementation_id == 'sklearn'


@pytest.mark.parametrize('metadata,device,environment', [
    (operation(GPU), 'cpu', ENVIRONMENT),
    (operation(GPU), 'cuda:0', replace(ENVIRONMENT, platform=ExecutionPlatform.windows)),
    (operation(CPU), 'cpu', replace(ENVIRONMENT, available_dependencies=frozenset())),
    (operation(), 'cpu', ENVIRONMENT),
])
def test_no_compatible_variant_is_reported_without_guessed_fallback(metadata, device, environment):
    with pytest.raises(ValueError, match='No available implementation.*example'):
        resolve(metadata, device=device, environment=environment)


def test_equally_suitable_variants_use_declaration_order():
    first = replace(GPU, id='first')
    second = replace(GPU, id='second')
    assert resolve(operation(first, second)).implementation_id == 'first'
    assert resolve(operation(second, first)).implementation_id == 'second'


def test_strategy_is_selected_by_task_and_variants_without_that_task_are_skipped():
    regression = TaskTypesEnum.regression
    reference = StrategyReference('not_imported.torch', 'Regressor')
    portable = replace(GPU, id='multi_task', strategies=((TASK, GPU_STRATEGY), (regression, reference)))
    metadata = operation(GPU, portable, tasks=(TASK, regression))

    result = resolve(metadata, task=regression)

    assert result.implementation_id == 'multi_task'
    assert result.strategy == reference


def test_globally_compatible_task_keeps_existing_operation_task_semantics():
    regression = TaskTypesEnum.regression
    reference = StrategyReference('not_imported.sklearn', 'Regressor')
    implementation = replace(CPU, strategies=((regression, reference),))
    result = resolve(operation(implementation, tasks=(regression,)), device='cpu',
                     task=TaskTypesEnum.ts_forecasting)
    assert result.strategy == reference


def test_incompatible_task_is_rejected():
    with pytest.raises(ValueError, match='No available implementation'):
        resolve(operation(CPU), task=TaskTypesEnum.clustering)


def test_selection_does_not_import_strategies_or_discover_hardware(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Pure selection must not import strategies or inspect actual CUDA availability')

    monkeypatch.setattr(StrategyReference, 'load', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)

    assert resolve(operation(CPU, GPU)).strategy == GPU_STRATEGY


def test_multi_output_uses_declared_cpu_capability_before_training():
    cpu = replace(CPU, supports_multi_output=True)
    result = resolve_operation_execution(operation(cpu, GPU), TASK, torch.device('cuda:1'), ENVIRONMENT,
                                         multi_output=True)
    assert result.implementation_id == 'sklearn'


def test_unsupported_multi_output_has_no_implicit_fallback():
    with pytest.raises(ValueError, match='No available implementation'):
        resolve_operation_execution(operation(GPU), TASK, torch.device('cuda:0'), ENVIRONMENT, multi_output=True)
