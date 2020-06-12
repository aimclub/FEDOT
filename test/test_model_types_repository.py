import os

from core.repository.model_types_repository import ModelTypesRepository
from core.repository.tasks import TaskTypesEnum


def mocked_path():
    test_data_path = str(os.path.dirname(__file__))
    repo_json_file_path = os.path.join(test_data_path, 'data/model_repository.json')
    return repo_json_file_path


def test_search_in_repository_by_tag_and_metainfo_correct():
    repo = ModelTypesRepository(mocked_path())

    model_names, _ = repo.suitable_model(task_type=TaskTypesEnum.regression,
                                         tags=['ml'])

    assert 'linear' in model_names
    assert len(model_names) == 3


def test_search_in_repository_by_id_correct():
    repo = ModelTypesRepository(mocked_path())

    model = repo.model_info_by_id(id='tpot')

    assert model.id == 'tpot'
    assert 'automl' in model.tags


def test_search_in_repository_by_tag_correct():
    repo = ModelTypesRepository(mocked_path())

    model_names, _ = repo.models_with_tag(tags=['automl'])
    assert 'tpot' in model_names
    assert len(model_names) == 1

    model_names, _ = repo.models_with_tag(tags=['simple', 'linear'], is_full_match=True)
    assert {'linear', 'logit', 'lasso', 'ridge'}.issubset(model_names)
    assert len(model_names) == 4

    model_names, _ = repo.models_with_tag(tags=['simple', 'linear'])
    assert {'linear', 'logit', 'knn', 'lda', 'lasso', 'ridge'}.issubset(model_names)
    assert len(model_names) == 6

    model_names, _ = repo.models_with_tag(tags=['non_real_tag'])
    assert len(model_names) == 0
