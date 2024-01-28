from typing import Dict, List, Optional

import numpy as np
from fedot.core.repository.operation_tags_n_repo_enums import DataOperationTagsEnum, ExcludedTagsEnum, \
    ModelTagsEnum, TagsEnum
from fedot.core.repository.operation_types_repo_enum import OperationMetaInfo, OperationReposEnum
from golem.core.log import default_log
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class OperationTypesRepository:
    """ Class for connecting models and data operations with json files with
        its descriptions and metadata
    """

    def __init__(self, operation_type: Optional[OperationReposEnum] = None):
        if operation_type is None:
            operation_type = OperationReposEnum.MODEL
        elif not isinstance(operation_type, OperationReposEnum):
            raise ValueError(f"Repository name should be OperationReposEnum, get {type(operation_type)} instead")

        self.log = default_log(self)
        self.operation_type = operation_type

    @property
    def repo(self):
        return self.operation_type.repo

    @property
    def operations(self):
        return self.repo

    @property
    def tags(self):
        return self.operation_type.tags

    @classmethod
    def get_available_repositories(cls):
        return list(OperationReposEnum.ALL)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # TODO add deleting repositories
        self.repo_path = None
        OperationTypesRepository.__repository_dict__[self.operation_type]['initialized_repo'] = None
        default_model_repo_file = OperationTypesRepository.__repository_dict__['model']['file']
        OperationTypesRepository.assign_repo('model', default_model_repo_file)

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.operation_type}"

    def operation_info_by_id(self, operation_id: str) -> Optional[OperationMetaInfo]:
        """ Get operation by it's name (id) """

        operation_id = get_operation_type_from_id(operation_id)

        for operation in self.repo:
            if operation.id == operation_id:
                # return first founded operation
                return operation
        self.log.warning(f'Operation {operation_id} not found in the repository')

    def suitable_operation(self,
                           task_type: Optional[TaskTypesEnum] = None,
                           data_type: Optional[DataTypesEnum] = None,
                           tags: Optional[List[TagsEnum]] = None,
                           is_full_match: bool = False,
                           forbidden_tags: Optional[List[TagsEnum]] = None,
                           forbidden_operations: List[str] = None) -> List[str]:
        """Method returns operations from repository for desired task and / or
        tags. Filtering method.

        Args:
            task_type: task to filter
            data_type: data type to filter
            tags: operations with which tags are required
            is_full_match: requires all tags to match, or at least one
            forbidden_tags: operations with such tags shouldn't be returned
            forbidden_operations: operation ids that should be excluded from result
        """

        forbidden_tags = forbidden_tags or list()
        forbidden_operations = set() if forbidden_operations is None else set(forbidden_operations)

        if not tags:
            forbidden_tags.extend(ExcludedTagsEnum)
            tags = list()

        if any(not isinstance(tag, TagsEnum) for tag in tags + forbidden_tags):
            raise ValueError(("Tag should be `TagsEnum`, get "
                              f"`{'`, `'.join(set(map(type, tags + forbidden_tags)))}` instead"))

        no_task = task_type is None
        operations_info = []
        for o in self.repo:
            is_desired_task = task_type in o.task_type or no_task
            tags_good = not tags or _is_operation_contains_tag(tags, o.tags, is_full_match)
            tags_bad = not forbidden_tags or not _is_operation_contains_tag(forbidden_tags, o.tags, False)
            if is_desired_task and tags_good and tags_bad:
                operations_info.append(o)

        if data_type:
            # ignore text and image data types: there are no operations with these `input_type`
            ignore_data_type = data_type in [DataTypesEnum.text, DataTypesEnum.image]
            if data_type == DataTypesEnum.ts:
                valid_data_types = [DataTypesEnum.ts, DataTypesEnum.table]
            else:
                valid_data_types = ensure_wrapped_in_sequence(data_type)
            if not ignore_data_type:
                operations_info = [o for o in operations_info if
                                   np.any([data_type in o.input_types for data_type in valid_data_types])]

        return [m.id for m in operations_info if m.id not in forbidden_operations]


def get_visualization_tags_map() -> Dict[str, List[str]]:
    """
    Returns map between repository tags and list of corresponding models for visualizations.
    """
    # Search for tags.
    allowed_tags = set([*ModelTagsEnum, *DataOperationTagsEnum])
    operations_map = {}
    for repo_name in OperationReposEnum:
        # get only single repositories
        if len(repo_name.value) == 1:
            repo = OperationTypesRepository(repo_name)
            for operation in repo.operations:
                for tag in set(operation.tags) & allowed_tags:
                    if tag not in operations_map:
                        operations_map[tag] = list()
                    operations_map[tag].append(operation.id)
    return operations_map


def _is_operation_contains_tag(candidate_tags: List[str],
                               operation_tags: List[str],
                               is_full_match: bool) -> bool:
    """The function checks which operations are suitable for the selected tags

    Args:
        candidate_tags: ``list`` with tags that the operation must have in order
            to fit the selected task
        operation_tags: ``list`` with tags with names as in repository json file
            which correspond to the considering operation
        is_full_match: requires all tags to match, or at least one

    Returns:
        bool: is there a match on the tags
    """

    matches = (tag in operation_tags for tag in candidate_tags)
    if is_full_match:
        return all(matches)
    else:
        return any(matches)


def atomized_model_type():
    return 'atomized_operation'


def atomized_model_meta_tags():
    return ['random'], ['any'], ['atomized']


def get_operations_for_task(task: Optional[Task] = None,
                            data_type: Optional[DataTypesEnum] = None,
                            operation_repo: Optional[OperationReposEnum] = None,
                            tags: List[TagsEnum] = None,
                            forbidden_tags: List[TagsEnum] = None,
                            forbidden_operations: List[str] = None):
    """Function returns aliases of operations.

    Args:
        task: task to solve
        data_type: type of input data
        operation_repo: repo with operations
        tags: tags for grabbing when filtering
        forbidden_tags: tags for skipping when filtering
        preset: operations from this preset will be obtained

    Returns:
        list:  operation aliases
    """
    task_type = task.task_type if task else None
    operation_repo = operation_repo or OperationReposEnum.DEFAULT

    repo = OperationTypesRepository(operation_repo)
    operations = repo.suitable_operation(task_type,
                                         data_type=data_type,
                                         tags=tags,
                                         forbidden_tags=forbidden_tags,
                                         forbidden_operations=forbidden_operations)
    return operations


def get_operation_type_from_id(operation_id):
    """
    Args:

        operation_id: operation name with optional postfix - text after / sign
    Returns:
        operation type - all characters before postfix (all characters if no postfix found)
    """

    splitted_operation_id = operation_id.split('/')
    if len(splitted_operation_id) > 2:
        raise ValueError(f'Incorrect number of postfixes in {operation_id}')
    return splitted_operation_id[0]
