import pytest
import torch

import fedot.api.api_utils.presets as presets_module
from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.operations import operation_execution as execution
from fedot.core.operations.operation_execution_rules import ExecutionEnvironment
from fedot.core.repository.operation_implementation import DeviceType, ExecutionPlatform


def test_new_operations_without_heavy_uses_pure_exclusion_rule():
    preset = OperationsPreset(Task(TaskTypesEnum.regression), 'best_quality')

    assert preset.new_operations_without_heavy(
        ['heavy'], ['light', 'heavy']) == ['light']


def test_filter_operations_by_preset_intersects_modification(monkeypatch):
    def fake_get_operations_for_task(task, data_type=None, mode='all', preset=None):
        if preset == 'best_quality':
            return ['rf', 'xgboost', 'knn']
        if preset == '*tree':
            return ['rf', 'xgboost']
        raise AssertionError(f'unexpected preset: {preset}')

    monkeypatch.setattr(
        presets_module, 'get_operations_for_task', fake_get_operations_for_task)

    preset = OperationsPreset(
        Task(TaskTypesEnum.classification), 'best_quality*tree')
    assert preset.filter_operations_by_preset() == ['rf', 'xgboost']


def test_removed_gpu_preset_explains_device_selection():
    preset = OperationsPreset(Task(TaskTypesEnum.classification), 'gpu')
    with pytest.raises(ValueError, match='TensorData on CUDA'):
        preset.filter_operations_by_preset()


@pytest.fixture
def available_dependencies(monkeypatch):
    metadata = OperationTypesRepository('model').operations
    requirements = frozenset(dependency for item in metadata
                              for variant in item.implementations for dependency in variant.requires)

    def snapshot(variants, device):
        return ExecutionEnvironment(ExecutionPlatform.linux, frozenset((DeviceType.cpu, DeviceType.cuda)), requirements)

    monkeypatch.setattr(execution, 'discover_execution_environment', snapshot)


def test_fast_train_keeps_its_logical_models_when_device_changes(available_dependencies):
    preset = OperationsPreset(Task(TaskTypesEnum.classification), 'fast_train')
    cpu = preset.filter_operations_by_preset(data_device=torch.device('cpu'))
    gpu = preset.filter_operations_by_preset(data_device=torch.device('cuda:0'))
    assert cpu == gpu
    assert {'rf', 'knn', 'logit'} <= set(cpu)
    assert 'minibatchsgd' not in gpu
    assert OperationTypesRepository('model').repository_name == 'model_repository.json'


def test_gpu_only_models_are_filtered_without_replacing_the_repository(available_dependencies):
    preset = OperationsPreset(Task(TaskTypesEnum.regression), 'best_quality')
    cpu = preset.filter_operations_by_preset(data_device=torch.device('cpu'))
    gpu = preset.filter_operations_by_preset(data_device=torch.device('cuda:0'))
    assert {'cd', 'elasticnet', 'mbsgdcregr'}.isdisjoint(cpu)
    assert {'cd', 'elasticnet', 'mbsgdcregr'} <= set(gpu)
    assert {'rfr', 'ridge', 'linear'} <= set(cpu) & set(gpu)


def test_explicit_operation_pool_is_checked_against_data_device(available_dependencies):
    preset = OperationsPreset(Task(TaskTypesEnum.regression), 'fast_train')
    result = preset.composer_params_based_on_preset(
        {'available_operations': ['cd', 'ridge']}, data_device=torch.device('cpu'))
    assert result['available_operations'] == ['ridge']
    with pytest.raises(ValueError, match='No operations are available'):
        preset.composer_params_based_on_preset({'available_operations': ['cd']}, data_device=torch.device('cpu'))


def test_multi_output_search_pool_excludes_single_target_gpu_only_models(available_dependencies):
    preset = OperationsPreset(Task(TaskTypesEnum.regression), 'best_quality')
    operations = preset.filter_available_operations(['cd', 'ridge', 'rfr'], torch.device('cuda:0'), multi_output=True)
    assert operations == ['ridge', 'rfr']


def test_removed_gpu_preset_is_rejected_even_with_explicit_operations():
    preset = OperationsPreset(Task(TaskTypesEnum.regression), 'gpu')
    with pytest.raises(ValueError, match='gpu preset has been removed'):
        preset.composer_params_based_on_preset({'available_operations': ['ridge']})
