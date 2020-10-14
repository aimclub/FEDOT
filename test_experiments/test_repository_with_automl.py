import os

from core.repository.model_types_repository import ModelTypesRepository


def mocked_path():
    test_data_path = str(os.path.dirname(__file__))
    repo_json_file_path = \
        os.path.join(test_data_path,
                     '..', 'test', 'data', 'model_repository_with_automl.json')
    return repo_json_file_path


def test_search_in_repository_with_automl_by_id_correct():
    repo = ModelTypesRepository(mocked_path())

    model = repo.model_info_by_id(id='tpot')

    assert model.id == 'tpot'
    assert 'automl' in model.tags


def test_search_in_repository_with_automl_by_tag_correct():
    repo = ModelTypesRepository(mocked_path())

    model_names, _ = repo.models_with_tag(tags=['automl'])
    assert 'tpot' in model_names
    assert len(model_names) == 1
