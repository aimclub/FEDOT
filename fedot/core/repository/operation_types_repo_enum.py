from collections import defaultdict
from enum import Enum
from itertools import chain
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional

from fedot.core.repository.operation_tags_n_repo_enums import ALL_TAGS, DataOperationTagsEnum, ModelTagsEnum, TagsEnum

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.json_evaluation import import_enums_from_str, import_strategy_from_str, read_field
from fedot.core.repository.tasks import Task, TaskTypesEnum

@dataclass
class OperationMetaInfo:
    id: str
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    allowed_positions: List[str]
    operation_types_repository: 'OperationReposEnum'
    strategies_json: str
    tags: Optional[List[TagsEnum]] = None
    presets: Optional[List[TagsEnum]] = None

    _supported_strategies = None

    @property
    def supported_strategies(self):
        if self._supported_strategies is None:
            if isinstance(self.strategies_json, list):
                supported_strategies = import_strategy_from_str(self.strategies_json)
            elif isinstance(self.strategies_json, dict):
                supported_strategies = dict()
                for strategy_dct_key, strategy_str_value in self.strategies_json.items():
                    import_path = import_enums_from_str(strategy_dct_key)
                    strategy_class = import_strategy_from_str(strategy_str_value)
                    supported_strategies[import_path] = strategy_class
            else:
                raise TypeError('strategies are of unknown type')
            self._supported_strategies = supported_strategies
        return self._supported_strategies

    def current_strategy(self, task: TaskTypesEnum) -> Optional['EvaluationStrategy']:
        """
        Gets available processing strategies depending on the selected task

        Args:
            task: machine learning task (e.g. regression and classification)

        Returns:
            supported strategies for task
        """

        if isinstance(self.supported_strategies, dict):
            return self.supported_strategies.get(task, None)
        return self.supported_strategies


class RepositoryFile:
    """ Load and store data from repository file """

    def __init__(self,
                 name: str,
                 file: Optional[str] = None):
        self.name = name
        self.file = Path(__file__).parent / 'data' / (file or f"{name}.json")
        self._repo = None
    
    def __str__(self) -> str:
        return self.name
    
    def __deepcopy__(self, memo=None):
        return self

    def __hash__(self):
        return (''.join(map(lambda x: x.id, self.repo))).__hash__()

    @property
    def tags(self) -> List[TagsEnum]:
        return list(set(chain(*[operation.tags for operation in self.repo])))
    
    @property
    def repo(self) -> List[OperationMetaInfo]:
        if self._repo is None:
            self._repo = _initialise_repo(self.file, repo_name=self.name)
        return self._repo


class _OperationRepoFilesEnum(Enum):
    MODEL = 'model_repository'
    GPU = 'gpu_models_repository'
    DATA_OPERATION = 'data_operation_repository'
    AUTOML = 'automl_repository'
    
    @property
    def tags(self):
        return REPOSITORY_BASE[self.value].tags
    
    @property
    def repo(self):
        return REPOSITORY_BASE[self.value].repo


REPOSITORY_BASE = {repo.value: RepositoryFile(name=repo.value) for repo in _OperationRepoFilesEnum}


class OperationReposEnum(Enum):
    DEFAULT = (_OperationRepoFilesEnum.MODEL, _OperationRepoFilesEnum.DATA_OPERATION)
    DEFAULT_GPU = (_OperationRepoFilesEnum.GPU, _OperationRepoFilesEnum.DATA_OPERATION)
    ALL = tuple(_OperationRepoFilesEnum)
    
    MODEL = (_OperationRepoFilesEnum.MODEL, )
    DATA_OPERATION = (_OperationRepoFilesEnum.DATA_OPERATION, )
    AUTOML = (_OperationRepoFilesEnum.AUTOML, )
    GPU = (_OperationRepoFilesEnum.GPU, )

    def __iter__(self):
        return self.value.__iter__()
    
    def __str__(self):
        return ', '.join(map(str, self))
    
    @property
    def tags(self):
        return list(chain(*(repo.tags for repo in self)))
    
    @property
    def repo(self):
        return list(chain(*(repo.repo for repo in self)))


def _initialise_repo(file: Path, repo_name: str) -> None:
        """ Method parse ``JSON`` repository with operations descriptions and
            wrapped information into :obj:`OperationMetaInfo`, then put it into the list in `self._repo`
        """
        
        repository_json = _load_repository(file)

        metadata_json = repository_json['metadata']
        operations_json = repository_json['operations']

        operations_list = []
        enum_tags = {tag.name: tag for tag in chain(*ALL_TAGS)}  # map from str tag to enum tag

        repo_map = {repo.value: next(crepo for crepo in OperationReposEnum
                                     if len(crepo.value) == 1 and crepo.value[0].value == repo.value)
                    for repo in _OperationRepoFilesEnum}
        for current_operation_key in operations_json:
            # Get information about operation
            # properties - information about operation by key, for example tags
            # metadata - information about meta of the operation
            properties = operations_json.get(current_operation_key)
            metadata = metadata_json[properties['meta']]

            task_types = import_enums_from_str(metadata['tasks'])
            input_type = import_enums_from_str(properties.get('input_type', metadata.get('input_type')))
            output_type = import_enums_from_str(properties.get('output_type', metadata.get('output_type')))

            # Get available strategies for obtained metadata
            strategies_json = metadata['strategies']

            accepted_node_types = read_field(metadata, 'accepted_node_types', ['any'])
            forbidden_node_types = read_field(metadata, 'forbidden_node_types', [])

            # Get tags for meta and for operation
            meta_tags = read_field(metadata, 'tags', [])
            operation_tags = read_field(properties, 'tags', [])
            presets = read_field(properties, 'presets', [])

            tags = list()
            for tag in set(meta_tags + operation_tags + presets):
                if tag in enum_tags:
                    tags.append(enum_tags[tag])
                else:
                    raise ValueError(f"Unknown tag {tag}")

            # Node position
            allowed_positions = ['primary', 'secondary', 'root']
            if accepted_node_types and accepted_node_types != 'all':
                allowed_positions = accepted_node_types
            if forbidden_node_types:
                allowed_positions = [pos for pos in allowed_positions if
                                     pos not in forbidden_node_types]

            operation = OperationMetaInfo(id=current_operation_key,
                                          input_types=input_type,
                                          output_types=output_type,
                                          task_type=task_types,
                                          strategies_json=strategies_json,
                                          allowed_positions=allowed_positions,
                                          tags=tags,
                                          presets=presets,
                                          operation_types_repository=repo_map[repo_name],
                                          )
            operations_list.append(operation)
        return operations_list


def _load_repository(repo_path: Path) -> dict:
    # Loads the repository for various cases and loads the necessary additional data "base_repository.json".
    with open(repo_path) as repository_json_file:
        repository_json = json.load(repository_json_file)

    if 'base_repository' in repository_json:
        base_repository_json_file = repo_path.with_name(repository_json['base_repository'])

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
