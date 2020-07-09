import os
import json
from core.repository.model_types_repository import ModelTypesRepository
from core.repository.tasks import TaskTypesEnum
from core.repository.json_evaluation import read_field, eval_field_str, eval_strategy_str
from core.models.evaluation.evaluation import SkLearnClassificationStrategy


def mocked_path():
    test_data_path = str(os.path.dirname(__file__))
    repo_json_file_path = os.path.join(test_data_path, 'data/model_repository.json')
    return repo_json_file_path


def test_lazy_load():
    repo = ModelTypesRepository(mocked_path())
    repo_second = ModelTypesRepository()

    assert repo._repo == repo_second._repo


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


def test_eval_field_str():
    model_metadata = _model_metadata_example(mocked_path())
    task_types = eval_field_str(model_metadata['tasks'])

    assert len(task_types) == 1
    assert task_types[0] == TaskTypesEnum.classification


def test_eval_strategy_str():
    model_metadata = _model_metadata_example(mocked_path())

    strategies_json = model_metadata['strategies']

    strategy = eval_strategy_str(strategies_json)
    assert strategy is SkLearnClassificationStrategy


def test_read_field():
    model_metadata = _model_metadata_example(mocked_path())
    meta_tags = read_field(model_metadata, 'tags', [])
    assert len(meta_tags) == 2
    assert 'ml' in meta_tags and 'sklearn' in meta_tags


def _model_metadata_example(path):
    with open(path) as repository_json_file:
        repository_json = json.load(repository_json_file)

    metadata_json = repository_json['metadata']
    models_json = repository_json['models']

    current_model_key = list(models_json.keys())[0]
    model_properties = [model_properties for model_key, model_properties in list(models_json.items())
                        if model_key == current_model_key][0]
    model_metadata = metadata_json[model_properties['meta']]
    return model_metadata
