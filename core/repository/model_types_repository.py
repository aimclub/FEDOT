from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import (
    List, Optional, Union)

from anytree import Node, RenderTree, findall

from core.repository.dataset_types import NumericalDataTypesEnum, DataTypesEnum, CategoricalDataTypesEnum
from core.repository.task_types import MachineLearningTasksEnum, TaskTypesEnum


class ModelTypesIdsEnum(Enum):
    xgboost = 'xgboost',
    knn = 'knn',
    logit = 'logit',
    dt = 'decisiontree',
    rf = 'randomforest',
    mlp = 'mlp',
    lda = 'lda',
    qda = 'qda'


class ModelGroupsIdsEnum(Enum):
    ml = 'ML_models'
    all = 'Models'


@dataclass
class ModelMetaInfo:
    input_type: List[DataTypesEnum]
    output_type: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    can_be_initial: bool = True
    can_be_secondary: bool = True


@dataclass
class ModelMetaInfoTemplate:
    input_type: DataTypesEnum = None
    output_type: DataTypesEnum = None
    task_type: TaskTypesEnum = None
    can_be_initial: bool = None
    can_be_secondary: bool = None

    @staticmethod
    def _is_field_suitable(candidate_field, template_field):
        if template_field is None:
            return True

        listed_candidate_field = candidate_field if isinstance(candidate_field, list) else [candidate_field]

        return template_field in listed_candidate_field

    def is_suits_for_template(self, candidate: ModelMetaInfo):
        fields = vars(self)

        fields_suitability = [self._is_field_suitable(getattr(candidate, field), getattr(self, field))
                              for field in fields]

        return all(fields_suitability)


class ModelsGroup(Node):
    def __init__(self, name: ModelGroupsIdsEnum, parent: Optional['ModelsGroup'] = None):
        super(Node, self).__init__()
        self.name = name
        self.parent = parent


class ModelType(Node):
    def __init__(self, name: ModelTypesIdsEnum, meta_info,
                 parent: Union[Optional['ModelsGroup'], Optional['ModelType']]):
        super(Node, self).__init__()
        self.name = name
        self.meta_info = meta_info
        self.parent = parent


class ModelTypesRepository:
    model_types = {type_: type_.value for type_ in ModelTypesIdsEnum}

    def _initialise_tree(self):
        root = ModelsGroup(ModelGroupsIdsEnum.all)

        ml = ModelsGroup(ModelGroupsIdsEnum.ml, parent=root)

        common_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                    output_type=[NumericalDataTypesEnum.vector, CategoricalDataTypesEnum.vector],
                                    task_type=[MachineLearningTasksEnum.classification,
                                               MachineLearningTasksEnum.regression])

        for model_type in ModelTypesIdsEnum:
            ModelType(model_type, deepcopy(common_meta), parent=ml)
        return root

    def __init__(self):
        self._tree = self._initialise_tree()

    def _is_in_path(self, node, desired_ids):
        return any(node_from_path.name in desired_ids for node_from_path in
                   node.path)

    def search_model_types_by_attributes(self,
                                         desired_ids: Optional[List[ModelGroupsIdsEnum]] = None,
                                         desired_metainfo: Optional[ModelMetaInfoTemplate] = None):

        desired_ids = [ModelGroupsIdsEnum.all] if desired_ids is None or not desired_ids else desired_ids

        results = findall(self._tree, filter_=lambda node: isinstance(node, ModelType) and
                                                           self._is_in_path(node, desired_ids))

        if desired_metainfo is not None:
            results = [result for result in results
                       if isinstance(result, ModelType) and
                       desired_metainfo.is_suits_for_template(result.meta_info)]

        return [result.name for result in results if (result.name in self.model_types)]

    def print_tree(self):
        for pre, node in RenderTree(self._tree):
            print(f'{pre}{node.name}')
