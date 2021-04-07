from fedot.api.api_utils import filter_models_by_preset
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task


def test_presets_classification():
    task = Task(TaskTypesEnum.classification)
    repo = OperationTypesRepository()

    class_models, _ = repo.suitable_operation(task.task_type)

    models_for_light_preset = filter_models_by_preset(task, 'light')
    models_for_ultra_light_preset = filter_models_by_preset(task, 'ultra_light')

    assert len(models_for_ultra_light_preset) < len(models_for_light_preset) < len(class_models)
    assert set(models_for_ultra_light_preset) == {'dt', 'logit', 'knn'}


def test_presets_regression():
    task = Task(TaskTypesEnum.regression)
    repo = OperationTypesRepository()
    regr_models, _ = repo.suitable_operation(task.task_type)

    models_for_light_preset = filter_models_by_preset(task, 'light')
    models_for_ultra_light_preset = filter_models_by_preset(task, 'ultra_light')

    assert len(models_for_ultra_light_preset) < len(models_for_light_preset) == len(regr_models)
    assert set(models_for_ultra_light_preset) == {'dtreg', 'lasso', 'ridge', 'linear'}
