from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import (List, Optional, Tuple, Union)

from anytree import Node, RenderTree, findall

from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import TaskTypesEnum


class ModelTypesIdsEnum(Enum):
    xgboost = 'xgboost'
    xgbreg = 'xgbreg'
    gbr = 'gradientregressor'
    adareg = 'adaregressor'
    sgdr = 'stochasticregressor'
    knnreg = 'knnregressor'
    knn = 'knn'
    logit = 'logit'
    dt = 'decisiontree'
    dtreg = 'decisiontreeregressor'
    treg = 'treeregressor'
    rf = 'randomforest',
    svc = 'linearsvc',
    svr = 'linearsvr'
    rfr = 'randomforestregressor',
    mlp = 'mlp',
    lda = 'lda',
    qda = 'qda',
    ar = 'ar',
    bernb = 'bernoullinb'
    arima = 'arima',
    lstm = 'lstm'
    linear = 'linear',
    ridge = 'ridge',
    lasso = 'lasso',
    elactic = 'elastic'
    kmeans = 'kmeans'
    tpot = 'tpot'
    h2o = 'h2o'
    direct_datamodel = 'data_model',  # a pseudo_model that allow injecting raw input data to the secondary nodes,
    diff_data_model = 'diff_data_model',  # model for scale-based decomposition
    additive_data_model = 'additive_model',
    trend_data_model = 'trend_data_model',
    residual_data_model = 'residual_data_model'


class ModelGroupsIdsEnum(Enum):
    ml = 'ML_models'
    data_models = 'data_models'
    decomposition_data_models = 'decomposition_data_models'
    composition_data_models = 'composition_data_models'

    stat = 'Stat_models'
    keras = 'Keras_models'
    all = 'Models'


@dataclass
class ModelMetaInfo:
    input_types: List[DataTypesEnum]
    output_types: List[DataTypesEnum]
    task_type: List[TaskTypesEnum]
    can_be_initial: bool = True
    can_be_secondary: bool = True
    is_affects_target: bool = False
    without_preprocessing: bool = False


@dataclass
class ModelMetaInfoTemplate:
    input_types: [DataTypesEnum] = None
    output_types: [DataTypesEnum] = None
    task_type: Union[List[TaskTypesEnum], TaskTypesEnum] = None
    can_be_initial: bool = None
    can_be_secondary: bool = None
    is_affects_target: bool = None
    without_preprocessing: bool = None

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
        keras = ModelsGroup(ModelGroupsIdsEnum.keras, parent=root)
        data_models = ModelsGroup(ModelGroupsIdsEnum.data_models, parent=root)
        decomposition_data_models = ModelsGroup(ModelGroupsIdsEnum.decomposition_data_models, parent=data_models)
        composition_data_models = ModelsGroup(ModelGroupsIdsEnum.data_models.composition_data_models,
                                              parent=data_models)

        self._initialise_models_group(models=[ModelTypesIdsEnum.arima, ModelTypesIdsEnum.ar],
                                      task_type=[TaskTypesEnum.ts_forecasting],
                                      input_types=[DataTypesEnum.ts],
                                      parent=stat)

        self._initialise_models_group(models=[ModelTypesIdsEnum.linear,
                                              ModelTypesIdsEnum.lasso,
                                              ModelTypesIdsEnum.ridge,
                                              ModelTypesIdsEnum.xgbreg,
                                              ModelTypesIdsEnum.adareg,
                                              ModelTypesIdsEnum.gbr,
                                              ModelTypesIdsEnum.knnreg,
                                              ModelTypesIdsEnum.dtreg,
                                              ModelTypesIdsEnum.treg,
                                              ModelTypesIdsEnum.rfr,
                                              ModelTypesIdsEnum.svr,
                                              ModelTypesIdsEnum.sgdr],
                                      task_type=[TaskTypesEnum.regression],
                                      input_types=[DataTypesEnum.table, DataTypesEnum.table.ts_lagged_table],
                                      output_types=[DataTypesEnum.table, DataTypesEnum.table.ts],
                                      parent=ml)

        self._initialise_models_group(models=[ModelTypesIdsEnum.rf,
                                              ModelTypesIdsEnum.dt,
                                              ModelTypesIdsEnum.mlp,
                                              ModelTypesIdsEnum.lda,
                                              ModelTypesIdsEnum.qda,
                                              ModelTypesIdsEnum.logit,
                                              ModelTypesIdsEnum.knn,
                                              ModelTypesIdsEnum.xgboost,
                                              ModelTypesIdsEnum.svc,
                                              ModelTypesIdsEnum.bernb],
                                      task_type=[TaskTypesEnum.classification],
                                      input_types=[DataTypesEnum.table],
                                      output_types=[DataTypesEnum.table],
                                      parent=ml)

        self._initialise_models_group(models=[ModelTypesIdsEnum.tpot, ModelTypesIdsEnum.h2o],
                                      task_type=[TaskTypesEnum.classification],
                                      parent=ml, is_initial=False, is_secondary=False)

        self._initialise_models_group(models=[ModelTypesIdsEnum.kmeans],
                                      task_type=[TaskTypesEnum.clustering],
                                      parent=stat)

        common_meta = ModelMetaInfo(input_types=[DataTypesEnum.table],
                                    output_types=[DataTypesEnum.table],
                                    task_type=[], can_be_initial=True)

        group_meta = deepcopy(common_meta)
        group_meta.input_types = [DataTypesEnum.table, DataTypesEnum.ts, DataTypesEnum.ts_lagged_table]
        group_meta.output_types = [DataTypesEnum.table, DataTypesEnum.ts, DataTypesEnum.ts_lagged_table]
        group_meta.task_type = [TaskTypesEnum.classification,
                                TaskTypesEnum.regression,
                                TaskTypesEnum.ts_forecasting]
        group_meta.without_preprocessing = True
        group_meta.can_be_initial = False
        ModelType(ModelTypesIdsEnum.direct_datamodel, deepcopy(group_meta), parent=data_models)

        group_meta = deepcopy(common_meta)
        group_meta.task_type = [TaskTypesEnum.ts_forecasting]
        group_meta.input_types = [DataTypesEnum.ts]
        group_meta.output_types = [DataTypesEnum.ts]
        group_meta.is_affects_target = True
        group_meta.without_preprocessing = True
        ModelType(ModelTypesIdsEnum.additive_data_model, deepcopy(group_meta), parent=composition_data_models)
        ModelType(ModelTypesIdsEnum.trend_data_model, deepcopy(group_meta), parent=decomposition_data_models)
        ModelType(ModelTypesIdsEnum.residual_data_model, deepcopy(group_meta), parent=decomposition_data_models)

        group_meta = deepcopy(common_meta)
        group_meta.task_type = [TaskTypesEnum.ts_forecasting]
        group_meta.input_types = [DataTypesEnum.ts_lagged_3d]
        group_meta.output_types = [DataTypesEnum.ts]
        ModelType(ModelTypesIdsEnum.lstm, deepcopy(group_meta), parent=keras)

        return root

    # TODO refactor
    def _initialise_models_group(self, models: List[ModelTypesIdsEnum],
                                 task_type: List[TaskTypesEnum],
                                 parent: ModelsGroup, is_initial=True, is_secondary=True,
                                 output_types=None, input_types=None):
        if not output_types:
            output_types = [DataTypesEnum.table]
        if not input_types:
            input_types = [DataTypesEnum.table]
        common_meta = ModelMetaInfo(input_types=input_types,
                                    output_types=output_types,
                                    task_type=[], can_be_initial=is_initial, can_be_secondary=is_secondary,
                                    is_affects_target=False)

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
