from fedot.api.api_utils import filter_models_by_preset
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


def test_presets_classification():
    repo = ModelTypesRepository()
    class_models, _ = repo.suitable_model(TaskTypesEnum.classification)

    models_for_light_preset = filter_models_by_preset(class_models, 'light')
    models_for_ultra_light_preset = filter_models_by_preset(class_models, 'ultra_light')

    assert len(models_for_ultra_light_preset) < len(models_for_light_preset) < len(class_models)
    assert set(models_for_ultra_light_preset) == {'dt', 'logit', 'knn'}


def test_presets_regression():
    repo = ModelTypesRepository()
    class_models, _ = repo.suitable_model(TaskTypesEnum.regression)

    models_for_light_preset = filter_models_by_preset(class_models, 'light')
    models_for_ultra_light_preset = filter_models_by_preset(class_models, 'ultra_light')

    assert len(models_for_ultra_light_preset) < len(models_for_light_preset) == len(class_models)
    assert set(models_for_ultra_light_preset) == {'dtreg', 'lasso', 'ridge', 'linear'}
