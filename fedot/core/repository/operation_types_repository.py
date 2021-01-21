import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional
from abc import abstractmethod

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.json_evaluation import eval_field_str, eval_strategy_str, read_field
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass
class ModelMetaInfo:
    # TODO add docstring
    id: str
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    supported_strategies: Any
    allowed_positions: List[str]
    tags: Optional[List[str]] = None

    def current_strategy(self, task: TaskTypesEnum):
        # TODO add docstring
        if isinstance(self.supported_strategies, dict):
            return self.supported_strategies.get(task, None)
        return self.supported_strategies


class OperationTypesRepository:
    """
    Базовый класс для предоставления информации об используемых моделях или
    операциях с данными и доступных для них типах задач

    """
    # TODO add docstring
    _repo = None

    def __init__(self, repo_path=None):
        if repo_path:
            OperationTypesRepository._repo = self._initialise_repo(repo_path)
        if not repo_path and not OperationTypesRepository._repo:
            self._set_repo_to_default_state()

        self._tags_excluded_by_default = ['non-default', 'expensive']

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._set_repo_to_default_state()

    @abstractmethod
    def _set_repo_to_default_state(self):
        raise NotImplementedError()

    @abstractmethod
    def _initialise_repo(self, repo_path: str) -> List[ModelMetaInfo]:
        raise NotImplementedError()

    @property
    def models(self):
        return OperationTypesRepository._repo

    def model_info_by_id(self, id: str) -> Optional[ModelMetaInfo]:
        # TODO add docstring
        models_with_id = [m for m in OperationTypesRepository._repo if m.id == id]
        if len(models_with_id) > 1:
            raise ValueError('Several models with same id in repository')
        if len(models_with_id) == 0:
            warnings.warn('Model {id} not found in the repository')
            return None
        return models_with_id[0]

    def models_with_tag(self, tags: List[str], is_full_match: bool = False):
        # TODO add docstring
        models_info = [m for m in OperationTypesRepository._repo if
                       _is_tags_contains_in_model(tags, m.tags, is_full_match)]
        return [m.id for m in models_info], models_info

    def suitable_model(self, task_type: TaskTypesEnum,
                       tags: List[str] = None, is_full_match: bool = False,
                       forbidden_tags: List[str] = None):
        # TODO add docstring

        if not forbidden_tags:
            forbidden_tags = []

        # TODO add comments
        for excluded_default_tag in self._tags_excluded_by_default:
            if not tags or excluded_default_tag not in tags:
                forbidden_tags.append(excluded_default_tag)

        # TODO add comments
        models_info = [m for m in OperationTypesRepository._repo if
                       task_type in m.task_type and
                       (not tags or _is_tags_contains_in_model(tags, m.tags,
                                                               is_full_match)) and
                       (not forbidden_tags or not _is_tags_contains_in_model(
                           forbidden_tags, m.tags, False))]
        return [m.id for m in models_info], models_info


class ModelTypesRepository(OperationTypesRepository):
    _repo = None

    def __init__(self, repo_path=None):
        super().__init__(repo_path)

    def _set_repo_to_default_state(self):
        # TODO add docstring
        repo_folder_path = str(os.path.dirname(__file__))
        file = 'data/model_repository.json'
        repo_path = os.path.join(repo_folder_path, file)
        OperationTypesRepository._repo = self._initialise_repo(repo_path)

    def _initialise_repo(self, repo_path: str) -> List[ModelMetaInfo]:
        # TODO add docstring
        with open(repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        metadata_json = repository_json['metadata']
        models_json = repository_json['models']

        models_list = []

        for current_model_key in models_json:
            # TODO add comments
            model_properties = \
            [model_properties for model_key, model_properties in
             list(models_json.items())
             if model_key == current_model_key][0]
            model_metadata = metadata_json[model_properties['meta']]

            task_types = eval_field_str(model_metadata['tasks'])
            input_type = eval_field_str(model_metadata['input_type'])
            output_type = eval_field_str(model_metadata['output_type'])

            strategies_json = model_metadata['strategies']
            if isinstance(strategies_json, list):
                supported_strategies = eval_strategy_str(strategies_json)
            else:
                supported_strategies = {}
                for strategy_dict_key in strategies_json.keys():
                    supported_strategies[
                        eval_field_str(strategy_dict_key)] = eval_strategy_str(
                        strategies_json[strategy_dict_key])

            accepted_node_types = read_field(model_metadata,
                                             'accepted_node_types', ['any'])
            forbidden_node_types = read_field(model_metadata,
                                              'forbidden_node_types', [])
            meta_tags = read_field(model_metadata, 'tags', [])

            model_tags = read_field(model_properties, 'tags', [])

            allowed_positions = ['primary', 'secondary', 'root']

            if accepted_node_types and accepted_node_types != 'all':
                allowed_positions = accepted_node_types
            if forbidden_node_types:
                allowed_positions = [pos for pos in allowed_positions if
                                     pos not in forbidden_node_types]

            tags = list(set(meta_tags + model_tags))

            model = ModelMetaInfo(id=current_model_key,
                                  input_types=input_type,
                                  output_types=output_type,
                                  task_type=task_types,
                                  supported_strategies=supported_strategies,
                                  allowed_positions=allowed_positions,
                                  tags=tags)
            models_list.append(model)

        return models_list


class DataOperationTypesRepository(OperationTypesRepository):
    _repo = None

    def __init__(self, repo_path=None):
        super().__init__(repo_path)

    def _set_repo_to_default_state(self):
        # TODO add docstring
        repo_folder_path = str(os.path.dirname(__file__))

        file = 'data/data_operation_repository.json'
        repo_path = os.path.join(repo_folder_path, file)
        OperationTypesRepository._repo = self._initialise_repo(repo_path)

    def _initialise_repo(self, repo_path: str) -> List[ModelMetaInfo]:
        # TODO add docstring
        with open(repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        metadata_json = repository_json['metadata']
        models_json = repository_json['data_operations']

        models_list = []

        for current_model_key in models_json:
            # TODO add comments
            model_properties = \
            [model_properties for model_key, model_properties in
             list(models_json.items())
             if model_key == current_model_key][0]
            model_metadata = metadata_json[model_properties['meta']]

            task_types = eval_field_str(model_metadata['tasks'])
            input_type = eval_field_str(model_metadata['input_type'])
            output_type = eval_field_str(model_metadata['output_type'])

            strategies_json = model_metadata['strategies']
            if isinstance(strategies_json, list):
                supported_strategies = eval_strategy_str(strategies_json)
            else:
                supported_strategies = {}
                for strategy_dict_key in strategies_json.keys():
                    supported_strategies[
                        eval_field_str(strategy_dict_key)] = eval_strategy_str(
                        strategies_json[strategy_dict_key])

            accepted_node_types = read_field(model_metadata,
                                             'accepted_node_types', ['any'])
            forbidden_node_types = read_field(model_metadata,
                                              'forbidden_node_types', [])
            meta_tags = read_field(model_metadata, 'tags', [])

            model_tags = read_field(model_properties, 'tags', [])

            allowed_positions = ['primary', 'secondary', 'root']

            if accepted_node_types and accepted_node_types != 'all':
                allowed_positions = accepted_node_types
            if forbidden_node_types:
                allowed_positions = [pos for pos in allowed_positions if
                                     pos not in forbidden_node_types]

            tags = list(set(meta_tags + model_tags))

            model = ModelMetaInfo(id=current_model_key,
                                  input_types=input_type,
                                  output_types=output_type,
                                  task_type=task_types,
                                  supported_strategies=supported_strategies,
                                  allowed_positions=allowed_positions,
                                  tags=tags)
            models_list.append(model)

        return models_list


def _is_tags_contains_in_model(candidate_tags: List[str], model_tags: List[str],
                               is_full_match: bool) -> bool:
    """
    The function checks which models are suitable for the selected tags

    :param candidate_tags: list with tags that the model must have in order
    to fit the selected task
    :param model_tags: list with tags with names as in repository json file
    which correspond to the considering model
    :param is_full_match: requires all tags to match, or at least one

    :return : is there a match on the tags
    """

    matches = ([(tag in model_tags) for tag in candidate_tags])
    if is_full_match:
        return all(matches)
    else:
        return any(matches)
