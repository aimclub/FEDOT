from fedot.api.api_utils.presets import OperationPreset
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_presets_classification():
    task = Task(TaskTypesEnum.classification)
    class_operations = get_operations_for_task(task=task, mode='all')

    preset_light = OperationPreset(task=task, preset='light')
    operations_for_light_preset = preset_light._filter_operations_by_preset()

    preset_ultra_light = OperationPreset(task=task, preset='ultra_light')
    operations_for_ultra_light_preset = preset_ultra_light._filter_operations_by_preset()

    assert len(operations_for_ultra_light_preset) < len(operations_for_light_preset) < len(class_operations)
    assert {'dt', 'logit', 'knn'} <= set(operations_for_ultra_light_preset)


def test_presets_regression():
    task = Task(TaskTypesEnum.regression)

    regr_operations = get_operations_for_task(task=task, mode='all')

    preset_light = OperationPreset(task=task, preset='light')
    operations_for_light_preset = preset_light._filter_operations_by_preset()
    preset_ultra_light = OperationPreset(task=task, preset='ultra_light')
    operations_for_ultra_light_preset = preset_ultra_light._filter_operations_by_preset()

    assert len(operations_for_ultra_light_preset) < len(operations_for_light_preset) <= len(regr_operations)
    assert {'dtreg', 'lasso', 'ridge', 'linear'} <= set(operations_for_ultra_light_preset)
