from unittest.mock import patch

from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import (ModelGroupsIdsEnum, ModelMetaInfo, ModelMetaInfoTemplate, ModelType,
                                                    ModelTypesIdsEnum, ModelTypesRepository, ModelsGroup)
from core.repository.tasks import Task, TaskTypesEnum


def default_mocked_tree():
    root = ModelsGroup(ModelGroupsIdsEnum.all)

    ml = ModelsGroup(ModelGroupsIdsEnum.ml, parent=root)

    xgboost_meta = ModelMetaInfo(
        input_types=[DataTypesEnum.table, DataTypesEnum.table],
        output_types=[DataTypesEnum.table, DataTypesEnum.table],
        task_type=[TaskTypesEnum.classification,
                   TaskTypesEnum.regression])

    ModelType(ModelTypesIdsEnum.xgboost, xgboost_meta, parent=ml)

    knn_meta = ModelMetaInfo(input_types=[DataTypesEnum.table],
                             output_types=[DataTypesEnum.table],
                             task_type=[TaskTypesEnum.classification])

    ModelType(ModelTypesIdsEnum.knn, knn_meta, parent=ml)

    logit_meta = ModelMetaInfo(
        input_types=[DataTypesEnum.table, DataTypesEnum.table],
        output_types=[DataTypesEnum.table],
        task_type=[TaskTypesEnum.classification])

    ModelType(ModelTypesIdsEnum.logit, logit_meta, parent=ml)

    return root


@patch('core.repository.model_types_repository.ModelTypesRepository._initialise_tree',
       side_effect=default_mocked_tree)
def test_search_in_repository_by_id_and_metainfo_correct(mock_init_tree):
    repo = ModelTypesRepository()

    model_names, _ = repo.search_models(
        desired_ids=[ModelGroupsIdsEnum.ml], desired_metainfo=ModelMetaInfoTemplate(
            task_type=TaskTypesEnum.regression))

    assert ModelTypesIdsEnum.xgboost in model_names
    assert len(model_names) == 1


@patch('core.repository.model_types_repository.ModelTypesRepository._initialise_tree',
       side_effect=default_mocked_tree)
def test_search_in_repository_by_model_id_correct(mock_init_tree):
    repo = ModelTypesRepository()

    model_names, _ = repo.search_models(
        desired_ids=[ModelGroupsIdsEnum.all])

    assert ModelTypesIdsEnum.xgboost in model_names
    assert len(model_names) == 3


@patch('core.repository.model_types_repository.ModelTypesRepository._initialise_tree',
       side_effect=default_mocked_tree)
def test_search_in_repository_by_metainfo_correct(mock_init_tree):
    repo = ModelTypesRepository()

    model_names, _ = repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_types=DataTypesEnum.table,
                                               output_types=DataTypesEnum.table,
                                               task_type=TaskTypesEnum.classification))

    assert ModelTypesIdsEnum.knn in model_names
    assert len(model_names) == 3
