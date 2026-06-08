import fedot.api.api_utils.presets as presets_module
from fedot.api.api_utils.presets import OperationsPreset
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _FakeRepository:
    def suitable_operation(self, task_type, data_type):
        return ['gpu_rf', 'gpu_logit']


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


def test_filter_operations_by_preset_uses_gpu_repository(monkeypatch):
    monkeypatch.setattr(
        presets_module.OperationTypesRepository,
        'assign_repo',
        staticmethod(lambda repo_name='model',
                     path='gpu_models_repository.json': _FakeRepository()),
    )

    preset = OperationsPreset(Task(TaskTypesEnum.classification), 'gpu')
    assert preset.filter_operations_by_preset() == ['gpu_logit', 'gpu_rf']
