from dataclasses import dataclass
from enum import Enum
from typing import (
    List, Optional, Union
)

from anytree import Node, RenderTree, findall

from core.evaluation import LogRegression
from core.evaluation import XGBoost
from core.repository.dataset_types import NumericalDataTypesEnum, DataTypesEnum, CategoricalDataTypesEnum
from core.repository.task_types import MachineLearningTasksEnum, TaskTypesEnum


class ModelTypesIdsEnum(Enum):
    xgboost = 'xgboost'
    knn = 'knn'
    logit = 'logit'


class ModelGroupsIdsEnum(Enum):
    ml = 'ML_models'
    all = 'Models'

@dataclass
class ModelMetaInfo:
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_types: List[TaskTypesEnum]
    can_be_initial: bool = True
    can_be_secondary: bool = True

@dataclass
class ModelMetaInfoTemplate:
    input_types: List[DataTypesEnum] = None
    output_types: List[DataTypesEnum] = None
    task_types: List[TaskTypesEnum] = None
    can_be_initial: bool = None
    can_be_secondary: bool = None

    @staticmethod
    def _is_candidate_field_suits_for_template_field(
            candidate_field: Optional[Union[List[DataTypesEnum], List[TaskTypesEnum],bool]],
            template_field: Optional[Union[List[DataTypesEnum], List[TaskTypesEnum],bool]]):
        if isinstance(candidate_field, list):
            return candidate_field is None or not candidate_field or \
                   template_field is None or not template_field or \
                   any([template_field in prop_item for prop_item in candidate_field])
        else:
            return template_field is None or not template_field or template_field == candidate_field

    def is_suits_for_template(self, candidate: ModelMetaInfo):
        fields = vars(self)
        return all(
            [self._is_candidate_field_suits_for_template_field(getattr(candidate, field), getattr(self, field))
             for field in fields])


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
    model_implementations = {
        ModelTypesIdsEnum.xgboost: XGBoost,
        ModelTypesIdsEnum.logit: LogRegression
    }

    def _initialise_tree(self):
        root = ModelsGroup(ModelGroupsIdsEnum.all)

        ml = ModelsGroup(ModelGroupsIdsEnum.ml, parent=root)

        xgboost_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                     output_types=[NumericalDataTypesEnum.vector, CategoricalDataTypesEnum.vector],
                                     task_types=[MachineLearningTasksEnum.classification,
                                                 MachineLearningTasksEnum.regression])

        ModelType(ModelTypesIdsEnum.xgboost, xgboost_meta, parent=ml)

        knn_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table],
                                 output_types=[CategoricalDataTypesEnum.vector],
                                 task_types=[MachineLearningTasksEnum.classification])

        ModelType(ModelTypesIdsEnum.knn, knn_meta, parent=ml)

        logit_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                   output_types=[CategoricalDataTypesEnum.vector],
                                   task_types=[MachineLearningTasksEnum.classification])

        ModelType(ModelTypesIdsEnum.logit, logit_meta, parent=ml)

        return root

    def __init__(self):
        self._tree = self._initialise_tree()

    def search_model_types_by_attributes(self,
                                         desired_ids: Optional[List[ModelGroupsIdsEnum]] = [ModelGroupsIdsEnum.all],
                                         desired_metainfo: Optional[ModelMetaInfoTemplate] = None):

        if desired_ids is None or not desired_ids:
            desired_ids = [ModelGroupsIdsEnum.all]

        results = findall(self._tree, filter_=lambda node: isinstance(node, ModelType) and
                                                           any(node_from_path.name in desired_ids for node_from_path in
                                                               node.path))

        if desired_metainfo is not None:
            results = [result for result in results
                       if (isinstance(result, ModelType) and desired_metainfo.is_suits_for_template(result.meta_info))]

        return [result.name for result in results]

    def obtain_model_implementation(self, model_type_id: ModelTypesIdsEnum):
        return self.model_implementations[model_type_id]()

    def print_tree(self):
        for pre, node in RenderTree(self._tree):
            print(f'{pre}{node.name}')
