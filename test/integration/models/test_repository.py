import json
import os

from fedot.core.operations.evaluation.classification import SkLearnClassificationStrategy
from fedot.core.repository.json_evaluation import import_enums_from_str, \
    import_strategy_from_str, read_field
from fedot.core.repository.operation_types_repository import (OperationTypesRepository,
                                                              get_operation_type_from_id)
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def mocked_path():
    test_data_path = str(os.path.dirname(__file__))
    repo_json_file_path = os.path.join(test_data_path, '../../data/model_repository.json')
    return repo_json_file_path


def test_lazy_load():
    with OperationTypesRepository() as repo:
        repo_second = OperationTypesRepository()

        assert repo._repo == repo_second._repo


def test_search_in_repository_by_tag_and_metainfo_correct():
    with OperationTypesRepository() as repo:
        model_names = repo.suitable_operation(task_type=TaskTypesEnum.regression,
                                              tags=['ml'])

        assert 'linear' in model_names
        assert len(model_names) == 17


def test_search_in_repository_by_tag_correct():
    with OperationTypesRepository() as repo:
        model_names = repo.operations_with_tag(tags=['simple', 'linear'], is_full_match=True)
        assert {'linear', 'logit', 'lasso', 'ridge'}.issubset(model_names)
        assert len(model_names) > 0

        model_names = repo.operations_with_tag(tags=['simple', 'linear'])
        assert {'linear', 'logit', 'knn', 'lda', 'lasso', 'ridge', 'polyfit'}.issubset(model_names)
        assert len(model_names) > 0

        model_names = repo.operations_with_tag(tags=['non_real_tag'])
        assert len(model_names) == 0


def test_eval_field_str():
    model_metadata = _model_metadata_example(mocked_path())
    task_types = import_enums_from_str(model_metadata['tasks'])

    assert len(task_types) == 1
    assert task_types[0] == TaskTypesEnum.classification


def test_eval_strategy_str():
    model_metadata = _model_metadata_example(mocked_path())

    strategies_json = model_metadata['strategies']

    strategy = import_strategy_from_str(strategies_json)
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
    models_json = repository_json['operations']

    current_model_key = list(models_json.keys())[0]
    model_properties = [model_properties for model_key, model_properties in list(models_json.items())
                        if model_key == current_model_key][0]
    model_metadata = metadata_json[model_properties['meta']]
    return model_metadata


def test_names_with_postfix():
    name_with_postfix = 'rf/best_model_ever'
    name_without_postfix = get_operation_type_from_id(name_with_postfix)
    assert name_without_postfix == 'rf'


def test_operation_types_repository_repr():
    repository = OperationTypesRepository().assign_repo('model', 'model_repository.json')

    assert repository.__repr__() == 'OperationTypesRepository for model_repository.json'


def test_repositories_tags_consistency():
    errors_found = []
    for repository in (OperationTypesRepository('model'), OperationTypesRepository('data_operation')):
        for operation in repository.operations:
            if repository.get_first_suitable_operation_tag(operation.id) is None:
                errors_found.append(f'{operation.id} in {repository} has no proper default tags!')

    assert not errors_found, '\n'.join(errors_found)


def test_pipeline_operation_repo_divide_operations():
    """ Checks whether the composer correctly divides operations into primary and secondary """

    available_operations = ['logit', 'rf', 'dt', 'xgboost']

    primary, secondary = \
        PipelineOperationRepository.divide_operations(task=Task(TaskTypesEnum.classification),
                                                      available_operations=available_operations)

    assert primary == available_operations
    assert secondary == available_operations
