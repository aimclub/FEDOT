import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.json_evaluation import eval_field_str, eval_strategy_str, read_field
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class OperationMetaInfo:
    id: str
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    supported_strategies: Any
    allowed_positions: List[str]
    tags: Optional[List[str]] = None

    def current_strategy(self, task: TaskTypesEnum):
        """
        Method allows getting available processing strategies depending on the
        selected task

        :param task: machine learning task (e.g. regression and classification)
        :return : supported strategies for task
        """

        if isinstance(self.supported_strategies, dict):
            return self.supported_strategies.get(task, None)
        return self.supported_strategies


def run_once(function):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return function(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class OperationTypesRepository:
    """ Class for connecting models and data operations with json files with
    its descriptions and metadata"""

    __initialized_repositories__ = {}

    __repository_dict__ = {
        'model': {'file': 'model_repository.json', 'initialized_repo': None},
        'data_operation': {'file': 'data_operation_repository.json', 'initialized_repo': None}
    }

    def __init__(self, operation_type: str = 'model'):
        self._tags_excluded_by_default = ['non-default', 'expensive']
        OperationTypesRepository.init_default_repositories()
        self.repository_name = OperationTypesRepository.__repository_dict__[operation_type]['file']
        self._repo = OperationTypesRepository.__repository_dict__[operation_type]['initialized_repo']

    @classmethod
    @run_once
    def init_default_repositories(cls):
        # default model repo
        default_model_repo_file = cls.__repository_dict__['model']['file']
        cls.assign_repo('model', default_model_repo_file)

        # default data_operation repo
        default_data_operation_repo_file = cls.__repository_dict__['data_operation']['file']
        cls.assign_repo('data_operation', default_data_operation_repo_file)

    @classmethod
    def assign_repo(cls, operation_type: str, repo_file: str):
        if operation_type not in ['model', 'data_operation']:
            raise Warning(f'The {operation_type} is not supported. The model type will be set')

        repo_path = create_repository_path(repo_file)
        if repo_file not in cls.__initialized_repositories__.keys():
            cls.__initialized_repositories__[repo_file] = cls._initialise_repo(repo_path)
        cls.__repository_dict__[operation_type]['file'] = repo_file
        cls.__repository_dict__[operation_type]['initialized_repo'] = cls.__initialized_repositories__[repo_file]

        return OperationTypesRepository(operation_type)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.repo_path = None

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.repository_name}"

    @classmethod
    def _initialise_repo(cls, repo_path: str) -> List[OperationMetaInfo]:
        """ Method parse JSON repository with operations descriptions and
        wrapped information into OperationMetaInfo, then put it into the list

        :return operations_list: list with OperationMetaInfo for every operation
        from json repository
        """
        repository_json = load_repository(repo_path)

        metadata_json = repository_json['metadata']
        operations_json = repository_json['operations']

        operations_list = []
        for current_operation_key in operations_json:
            # Get information about operation
            # properties - information about operation by key, for example tags
            # metadata - information about meta of the operation
            properties = operations_json.get(current_operation_key)
            metadata = metadata_json[properties['meta']]

            task_types = eval_field_str(metadata['tasks'])
            input_type = eval_field_str(properties['input_type']) \
                if ('input_type' in properties) \
                else eval_field_str(metadata['input_type'])
            output_type = eval_field_str(properties['output_type']) \
                if ('output_type' in properties) \
                else eval_field_str(metadata['output_type'])

            # Get available strategies for obtained metadata
            supported_strategies = OperationTypesRepository.get_strategies_by_metadata(metadata)

            accepted_node_types = read_field(metadata, 'accepted_node_types', ['any'])
            forbidden_node_types = read_field(metadata, 'forbidden_node_types', [])

            # Get tags for meta and for operation
            meta_tags = read_field(metadata, 'tags', [])
            operation_tags = read_field(properties, 'tags', [])

            allowed_positions = ['primary', 'secondary', 'root']

            if accepted_node_types and accepted_node_types != 'all':
                allowed_positions = accepted_node_types
            if forbidden_node_types:
                allowed_positions = [pos for pos in allowed_positions if
                                     pos not in forbidden_node_types]

            # Unit tags
            tags = meta_tags + operation_tags

            operation = OperationMetaInfo(id=current_operation_key,
                                          input_types=input_type,
                                          output_types=output_type,
                                          task_type=task_types,
                                          supported_strategies=supported_strategies,
                                          allowed_positions=allowed_positions,
                                          tags=tags)
            operations_list.append(operation)

        return operations_list

    @staticmethod
    def get_strategies_by_metadata(metadata: dict):
        """ Method allow obtain strategy instance by the metadata

        :param metadata: information about meta of the operation
        :return supported_strategies: available strategies for current metadata
        """
        strategies_json = metadata['strategies']
        if isinstance(strategies_json, list):
            supported_strategies = eval_strategy_str(strategies_json)
        else:
            supported_strategies = {}
            for strategy_dict_key in strategies_json.keys():
                # Convert string into class path for import
                import_path = eval_field_str(strategy_dict_key)
                strategy_class = eval_strategy_str(strategies_json[strategy_dict_key])

                supported_strategies.update({import_path: strategy_class})
        return supported_strategies

    def operation_info_by_id(self, operation_id: str) -> Optional[OperationMetaInfo]:
        """ Get operation by it's name (id) """

        operation_id = get_operation_type_from_id(operation_id)

        operations_with_id = [m for m in self._repo if m.id == operation_id]
        if len(operations_with_id) > 1:
            raise ValueError('Several operations with same id in repository')
        if len(operations_with_id) == 0:
            warnings.warn(f'Operation {operation_id} not found in the repository')
            return None
        return operations_with_id[0]

    def operations_with_tag(self, tags: List[str], is_full_match: bool = False):
        operations_info = [m for m in self._repo if
                           _is_tags_contains_in_operation(tags, m.tags, is_full_match)]
        return [m.id for m in operations_info], operations_info

    def suitable_operation(self, task_type: TaskTypesEnum = None,
                           data_type: TaskTypesEnum = None,
                           tags: List[str] = None, is_full_match: bool = False,
                           forbidden_tags: List[str] = None):
        """ Method returns operations from repository for desired task and / or
        tags. Filtering method.

        :param task_type: task filter
        :param tags: operations with which tags are required
        :param is_full_match: requires all tags to match, or at least one
        :param forbidden_tags: operations with such tags shouldn't be returned
        """

        if not forbidden_tags:
            forbidden_tags = []

        if not tags:
            for excluded_default_tag in self._tags_excluded_by_default:
                # Forbidden tags by default
                forbidden_tags.append(excluded_default_tag)

        if task_type is None:
            operations_info = [m for m in self._repo if
                               (not tags or _is_tags_contains_in_operation(tags, m.tags, is_full_match)) and
                               (not forbidden_tags or not _is_tags_contains_in_operation(forbidden_tags, m.tags,
                                                                                         False))]
        else:
            operations_info = [m for m in self._repo if task_type in m.task_type and
                               (not tags or _is_tags_contains_in_operation(tags, m.tags, is_full_match)) and
                               (not forbidden_tags or not _is_tags_contains_in_operation(forbidden_tags, m.tags,
                                                                                         False))]

        if data_type:
            operations_info = [o for o in operations_info if data_type in o.input_types]

        return [m.id for m in operations_info], operations_info

    @property
    def operations(self):
        return self._repo


def _is_tags_contains_in_operation(candidate_tags: List[str],
                                   operation_tags: List[str],
                                   is_full_match: bool) -> bool:
    """
    The function checks which operations are suitable for the selected tags

    :param candidate_tags: list with tags that the operation must have in order
    to fit the selected task
    :param operation_tags: list with tags with names as in repository json file
    which correspond to the considering operation
    :param is_full_match: requires all tags to match, or at least one

    :return : is there a match on the tags
    """

    matches = ([(tag in operation_tags) for tag in candidate_tags])
    if is_full_match:
        return all(matches)
    else:
        return any(matches)


def atomized_model_type():
    return 'atomized_operation'


def atomized_model_meta_tags():
    return ['random'], ['any'], ['atomized']


def get_operations_for_task(task: Optional[Task], mode='all', tags=None, forbidden_tags=None, ):
    """ Function returns aliases of operations.

    :param task: task to solve
    :param mode: mode to return operations
        The possible parameters are:
            'all' - return list with all operations
            'model' - return only list with models
            'data_operation' - return only list with data_operations
    :param tags: tags for grabbing when filtering
    :param forbidden_tags: tags for skipping when filtering

    :return : list with operation aliases
    """
    task_type = task.task_type if task else None
    if mode != 'all':
        model_types, _ = OperationTypesRepository(mode). \
            suitable_operation(task_type, tags=tags, forbidden_tags=forbidden_tags)
        return model_types
    elif mode == 'all':
        # Get models from repository
        model_types, _ = OperationTypesRepository('model') \
            .suitable_operation(task_type, tags=tags, forbidden_tags=forbidden_tags)
        # Get data operations
        data_operation_types, _ = OperationTypesRepository('data_operation') \
            .suitable_operation(task_type, tags=tags, forbidden_tags=forbidden_tags)
        return model_types + data_operation_types
    else:
        raise ValueError(f'Such mode "{mode}" is not supported')


def get_operation_type_from_id(operation_id):
    operation_type = _operation_name_without_postfix(operation_id)
    return operation_type


def _operation_name_without_postfix(operation_id):
    """
    :param operation_id: operation name with optional postfix - text after / sign
    :return: operation type - all characters before postfix (all characters if no postfix found)
    """
    postfix_sign = '/'
    # if the operation id has custom postfix
    if postfix_sign in operation_id:
        if operation_id.count(postfix_sign) > 1:
            raise ValueError(f'Incorrect number of postfixes in {operation_id}')
        return operation_id.split('/')[0]
    else:
        return operation_id


def load_repository(repo_path: str) -> dict:
    # Loads the repository for various cases and loads the necessary additional data "base_repository.json".
    with open(repo_path) as repository_json_file:
        repository_json = json.load(repository_json_file)

    if 'base_repository' in repository_json:
        base_repository_json_file = create_repository_path(repository_json['base_repository'])

        with open(base_repository_json_file) as repository_json_file:
            base_repository_json = json.load(repository_json_file)

        merged_dict = defaultdict(dict)

        merged_dict.update(base_repository_json)
        for key, nested_dict in repository_json.items():
            if key not in merged_dict:
                merged_dict[key] = nested_dict
            else:
                merged_dict[key].update(nested_dict)

        repository_json = dict(merged_dict)

    return repository_json


def create_repository_path(repository_name: str) -> str:
    # Path till repository file
    file = os.path.join('data', repository_name)
    # Path till current file with script
    repo_folder_path = str(os.path.dirname(__file__))
    repo_path = os.path.join(repo_folder_path, file)

    return repo_path
