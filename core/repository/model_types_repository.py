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
    svc = 'linearsvc',
    mlp = 'mlp',
    lda = 'lda',
    qda = 'qda',
    ar = 'ar',
    arima = 'arima',
    linear = 'linear',
    ridge = 'ridge',
    lasso = 'lasso',
    kmeans = 'kmeans'
    tpot = 'tpot'
    h2o = 'h2o'


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
    task_type: Union[List[TaskTypesEnum], TaskTypesEnum] = None
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

        self._initialise_models_group(models=[ModelTypesIdsEnum.arima, ModelTypesIdsEnum.ar],
                                      task_type=[MachineLearningTasksEnum.auto_regression],
                                      parent=stat)

        self._initialise_models_group(models=[ModelTypesIdsEnum.linear,
                                              ModelTypesIdsEnum.lasso,
                                              ModelTypesIdsEnum.ridge],
                                      task_type=[MachineLearningTasksEnum.regression],
                                      parent=ml)

        self._initialise_models_group(models=[ModelTypesIdsEnum.rf,
                                              ModelTypesIdsEnum.dt,
                                              ModelTypesIdsEnum.mlp,
                                              ModelTypesIdsEnum.lda,
                                              ModelTypesIdsEnum.qda,
                                              ModelTypesIdsEnum.logit,
                                              ModelTypesIdsEnum.knn,
                                              ModelTypesIdsEnum.xgboost,
                                              ModelTypesIdsEnum.svc],
                                      task_type=[MachineLearningTasksEnum.classification],
                                      parent=ml)

        self._initialise_models_group(models=[ModelTypesIdsEnum.tpot, ModelTypesIdsEnum.h2o],
                                      task_type=[MachineLearningTasksEnum.classification],
                                      parent=ml, is_initial=False, is_secondary=False)

        self._initialise_models_group(models=[ModelTypesIdsEnum.kmeans],
                                      task_type=[MachineLearningTasksEnum.clustering],
                                      parent=stat)


        return root

    def _initialise_models_group(self, models: List[ModelTypesIdsEnum],
                                 task_type: List[MachineLearningTasksEnum],
                                 parent: ModelsGroup, is_initial=True, is_secondary=True):

        common_meta = ModelMetaInfo(input_type=[NumericalDataTypesEnum.table, CategoricalDataTypesEnum.table],
                                    output_type=[NumericalDataTypesEnum.vector, CategoricalDataTypesEnum.vector],
                                    task_type=[], can_be_initial=is_initial, can_be_secondary=is_secondary)
        group_meta = deepcopy(common_meta)
        group_meta.task_type = task_type

        for model_type in models:
            ModelType(model_type, deepcopy(group_meta), parent=parent)

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

        models_ids = [result.name for result in results if (result.name in self.model_types)]
        models_metainfo = [result.meta_info for result in results if (result.name in self.model_types)]

        return models_ids, models_metainfo

    def print_structure(self):
        for pre, node in RenderTree(self._tree):
            print(f'{pre}{node.name}')
