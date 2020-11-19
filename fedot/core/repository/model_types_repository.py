import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.json_evaluation import eval_field_str, eval_strategy_str, read_field
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass
class ModelMetaInfo:
    id: str
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    supported_strategies: Any
    allowed_positions: List[str]
    tags: Optional[List[str]] = None

    def current_strategy(self, task: TaskTypesEnum):
        if isinstance(self.supported_strategies, dict):
            return self.supported_strategies.get(task, None)
        return self.supported_strategies


class ModelTypesRepository:
    _repo = None

    def __init__(self, repo_path=None):
        if repo_path:
            ModelTypesRepository._repo = self._initialise_repo(repo_path)
        if not repo_path and not ModelTypesRepository._repo:
            self._set_repo_to_default_state()

        self._tags_excluded_by_default = ['non-default', 'expensive']

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._set_repo_to_default_state()

    def _set_repo_to_default_state(self):
        repo_folder_path = str(os.path.dirname(__file__))
        file = 'data/model_repository.json'
        repo_path = os.path.join(repo_folder_path, file)
        ModelTypesRepository._repo = self._initialise_repo(repo_path)

    def _initialise_repo(self, repo_path: str) -> List[ModelMetaInfo]:
        with open(repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

        metadata_json = repository_json['metadata']
        models_json = repository_json['models']

        models_list = []

        for current_model_key in models_json:
            model_properties = [model_properties for model_key, model_properties in list(models_json.items())
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
                    supported_strategies[eval_field_str(strategy_dict_key)] = eval_strategy_str(
                        strategies_json[strategy_dict_key])

            accepted_node_types = read_field(model_metadata, 'accepted_node_types', ['any'])
            forbidden_node_types = read_field(model_metadata, 'forbidden_node_types', [])
            meta_tags = read_field(model_metadata, 'tags', [])

            model_tags = read_field(model_properties, 'tags', [])

            allowed_positions = ['primary', 'secondary', 'root']

            if accepted_node_types and accepted_node_types != 'all':
                allowed_positions = accepted_node_types
            if forbidden_node_types:
                allowed_positions = [pos for pos in allowed_positions if pos not in forbidden_node_types]

            tags = list(set(meta_tags + model_tags))

            model = ModelMetaInfo(id=current_model_key,
                                  input_types=input_type, output_types=output_type, task_type=task_types,
                                  supported_strategies=supported_strategies,
                                  allowed_positions=allowed_positions,
                                  tags=tags)
            models_list.append(model)

        return models_list

    @property
    def models(self):
        return ModelTypesRepository._repo

    def model_info_by_id(self, id: str) -> Optional[ModelMetaInfo]:
        models_with_id = [m for m in ModelTypesRepository._repo if m.id == id]
        if len(models_with_id) > 1:
            raise ValueError('Several models with same id in repository')
        if len(models_with_id) == 0:
            warnings.warn('Model {id} not found in the repository')
            return None
        return models_with_id[0]

    def models_with_tag(self, tags: List[str], is_full_match: bool = False):
        models_info = [m for m in ModelTypesRepository._repo if
                       _is_tags_contains_in_model(tags, m.tags, is_full_match)]
        return [m.id for m in models_info], models_info

    def suitable_model(self, task_type: TaskTypesEnum,
                       tags: List[str] = None, is_full_match: bool = False,
                       forbidden_tags: List[str] = None):

        if not forbidden_tags:
            forbidden_tags = []

        for excluded_default_tag in self._tags_excluded_by_default:
            if not tags or excluded_default_tag not in tags:
                forbidden_tags.append(excluded_default_tag)

        models_info = [m for m in ModelTypesRepository._repo if task_type in m.task_type and
                       (not tags or _is_tags_contains_in_model(tags, m.tags, is_full_match)) and
                       (not forbidden_tags or not _is_tags_contains_in_model(forbidden_tags, m.tags, False))]
        return [m.id for m in models_info], models_info


def _is_tags_contains_in_model(candidate_tags: List[str], model_tags: List[str], is_full_match: bool):
    matches = ([(tag in model_tags) for tag in candidate_tags])
    if is_full_match:
        return all(matches)
    else:
        return any(matches)
