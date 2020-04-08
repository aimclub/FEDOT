from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import (
    List, Optional, Union, Tuple)

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
    qda = 'qda',
    ar = 'ar',
    arima = 'arima',
    linear = 'linear',
    ridge = 'ridge',
    lasso = 'lasso',
    kmeans = 'kmeans'


class ModelGroupsIdsEnum(Enum):
    ml = 'ML_models'
    stat = 'Stat_models'
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
    task_type: Union[List[TaskTypesEnum],TaskTypesEnum] = None
    can_be_initial: bool = None
    can_be_secondary: bool = None

    @staticmethod
    def _is_field_suitable(candidate_field, template_field) -> bool:
        if template_field is None:
            return True

        listed_candidate_field = candidate_field if isinstance(candidate_field, list) else [candidate_field]

        if isinstance(template_field, list):
            return any([template_field_item in listed_candidate_field for template_field_item in template_field])
        else:
            return template_field in listed_candidate_field

    def is_suits_for_template(self, candidate: ModelMetaInfo) -> bool:
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
        stat = ModelsGroup(ModelGroupsIdsEnum.stat, parent=root)

        common_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                    output_type=[NumericalDataTypesEnum.vector, CategoricalDataTypesEnum.vector],
                                    task_type=[MachineLearningTasksEnum.classification,
                                               MachineLearningTasksEnum.regression])

        ar_meta = deepcopy(common_meta)
        ar_meta.task_type = [MachineLearningTasksEnum.auto_regression]

        reg_meta = deepcopy(common_meta)
        reg_meta.task_type = [MachineLearningTasksEnum.regression]

        class_meta = deepcopy(common_meta)
        class_meta.task_type = [MachineLearningTasksEnum.classification]

        clust_meta = deepcopy(common_meta)
        clust_meta.task_type = [MachineLearningTasksEnum.clustering]

        for model_type in ModelTypesIdsEnum:
            if model_type in [ModelTypesIdsEnum.arima,
                              ModelTypesIdsEnum.ar]:
                ModelType(model_type, deepcopy(ar_meta), parent=stat)
            elif model_type in [ModelTypesIdsEnum.linear,
                                ModelTypesIdsEnum.lasso,
                                ModelTypesIdsEnum.ridge]:
                ModelType(model_type, deepcopy(reg_meta), parent=ml)
            elif model_type in [ModelTypesIdsEnum.rf,
                                ModelTypesIdsEnum.dt,
                                ModelTypesIdsEnum.mlp,
                                ModelTypesIdsEnum.lda,
                                ModelTypesIdsEnum.qda,
                                ModelTypesIdsEnum.logit,
                                ModelTypesIdsEnum.knn,
                                ModelTypesIdsEnum.xgboost]:
                ModelType(model_type, deepcopy(class_meta), parent=ml)
            elif model_type in [ModelTypesIdsEnum.kmeans]:
                ModelType(model_type, deepcopy(clust_meta), parent=stat)
            else:
                ModelType(model_type, deepcopy(common_meta), parent=ml)

        return root

    def __init__(self):
        self._tree = self._initialise_tree()

    def _is_in_path(self, node, desired_ids):
        return any(node_from_path.name in desired_ids for node_from_path in
                   node.path)

    def search_models(self,
                      desired_ids:
                      Optional[List[Union[ModelGroupsIdsEnum, ModelTypesIdsEnum]]] = None,
                      desired_metainfo:
                      Optional[ModelMetaInfoTemplate] = None) -> Tuple[List[ModelTypesIdsEnum], List[ModelMetaInfo]]:

        desired_ids = [ModelGroupsIdsEnum.all] if desired_ids is None or not desired_ids else desired_ids

        results = findall(self._tree, filter_=lambda node: isinstance(node, ModelType) and
                                                           self._is_in_path(node, desired_ids))

        if desired_metainfo is not None:
            results = [result for result in results
                       if isinstance(result, ModelType) and
                       desired_metainfo.is_suits_for_template(result.meta_info)]

        return ([result.name for result in results if (result.name in self.model_types)],
                [result.meta_info for result in results if (result.name in self.model_types)])


def print_tree(self):
    for pre, node in RenderTree(self._tree):
        print(f'{pre}{node.name}')
