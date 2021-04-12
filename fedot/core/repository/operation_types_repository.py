import json
import os
import warnings
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


class OperationTypesRepository:
    """ Class for connecting models and data operations with json files with
    its descriptions and metadata"""
    _repo = None

    def __init__(self, repository_name: str = 'model_repository.json'):
        # Path till current file with script
        repo_folder_path = str(os.path.dirname(__file__))
        # Path till repository file
        file = os.path.join('data', repository_name)
        repo_path = os.path.join(repo_folder_path, file)
        self._repo = self._initialise_repo(repo_path)

        self._tags_excluded_by_default = ['non-default', 'expensive']

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.repo_path = None

    def _initialise_repo(self, repo_path: str) -> List[OperationMetaInfo]:
        """ Method parse JSON repository with operations descriptions and
        wrapped information into OperationMetaInfo, then put it into the list

        :return operations_list: list with OperationMetaInfo for every operation
        from json repository
        """
        with open(repo_path) as repository_json_file:
            repository_json = json.load(repository_json_file)

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
            input_type = eval_field_str(metadata['input_type'])
            output_type = eval_field_str(metadata['output_type'])

            # Get available strategies for obtained metadata
            supported_strategies = self.get_strategies_by_metadata(metadata)

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

        for excluded_default_tag in self._tags_excluded_by_default:
            if not tags or excluded_default_tag not in tags:
                # Forbidden tags by default
                forbidden_tags.append(excluded_default_tag)

        if task_type is None:
            operations_info = [m for m in self._repo if (not tags or _is_tags_contains_in_operation(tags, m.tags, is_full_match)) and
                               (not forbidden_tags or not _is_tags_contains_in_operation(forbidden_tags, m.tags, False))]
        else:
            operations_info = [m for m in self._repo if task_type in m.task_type and
                               (not tags or _is_tags_contains_in_operation(tags, m.tags, is_full_match)) and
                               (not forbidden_tags or not _is_tags_contains_in_operation(forbidden_tags, m.tags, False))]
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


def get_operations_for_task(task: Task, mode='all'):
    """ Function returns aliases of operations. Simplify OperationTypesRepository
    logic, but there are no ability to use tags for filtering

    :param task: task to solve
    :param mode: mode to return operations
        The possible parameters are:
            'all' - return list with all operations
            'models' - return only list with models
            'data_operations' - return only list with data_operations
    :return : list with operation aliases
    """

    # Get models from repository
    model_types, _ = OperationTypesRepository('model_repository.json')\
        .suitable_operation(task.task_type)
    # Get data operations
    data_operation_types, _ = OperationTypesRepository('data_operation_repository.json')\
        .suitable_operation(task.task_type)

    # Unit two lists
    available_operations = model_types + data_operation_types

    if mode == 'all':
        return available_operations
    elif mode == 'models':
        return model_types
    elif mode == 'data_operations':
        return data_operation_types
    else:
        raise ValueError(f'For "{mode}" there are no operations')


def get_ts_operations(tags=None, forbidden_tags=None, mode='all'):
    """ Function returns operations names for time series forecasting task

    :param tags: tags for grabbing when filtering
    :param forbidden_tags: tags for skipping when filtering
    :param mode: available modes 'models', 'data_operations' and 'all'
    """
    models_repo = OperationTypesRepository()
    models, _ = models_repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
                                               tags=tags, forbidden_tags=forbidden_tags)

    data_operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
    data_operations, _ = data_operations_repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
                                                                 tags=tags, forbidden_tags=forbidden_tags)

    if mode == 'models':
        return models
    elif mode == 'data_operations':
        return data_operations
    elif mode == 'all':
        # Unit two lists
        ts_operations = models + data_operations
        return ts_operations
    else:
        raise ValueError(f'Such mode "{mode}" is not supported')
