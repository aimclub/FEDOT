from core.evaluation import XGBoost, LogRegression
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import ModelTypesRepository, ModelGroupsIdsEnum, ModelMetaInfoTemplate, \
    ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum


def test_search_in_repository_by_id_and_metainfo():
    repo = ModelTypesRepository()

    model_names = repo.search_model_types_by_attributes(
        desired_ids=[ModelGroupsIdsEnum.ml],
        desired_metainfo=ModelMetaInfoTemplate(task_types=MachineLearningTasksEnum.regression))

    impl = repo.obtain_model_implementation(model_names[0])

    assert model_names[0] is ModelTypesIdsEnum.xgboost
    assert len(model_names) == 1
    assert isinstance(impl, XGBoost)


def test_search_in_repository_by_model_id():
    repo = ModelTypesRepository()

    model_names = repo.search_model_types_by_attributes(desired_ids=[ModelGroupsIdsEnum.all])
    impl = repo.obtain_model_implementation(model_names[0])

    assert model_names[0] is ModelTypesIdsEnum.xgboost
    assert len(model_names) == 3
    assert isinstance(impl, XGBoost)


def test_search_in_repository_by_metainfo():
    repo = ModelTypesRepository()

    model_names = repo.search_model_types_by_attributes(desired_metainfo=ModelMetaInfoTemplate(
        input_types=NumericalDataTypesEnum.table,
        output_types=CategoricalDataTypesEnum.vector,
        task_types=MachineLearningTasksEnum.classification))
    impl = repo.obtain_model_implementation(model_names[2])

    assert model_names[2] is ModelTypesIdsEnum.logit
    assert len(model_names) == 3
    assert isinstance(impl, LogRegression)


def test_direct_model_query():
    repo = ModelTypesRepository()

    impl = repo.obtain_model_implementation(ModelTypesIdsEnum.logit)

    assert isinstance(impl, LogRegression)
