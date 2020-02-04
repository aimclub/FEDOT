from anytree import Node, RenderTree, findall
from enum import Enum
from core.evaluation import XGBoost
from core.evaluation import LogRegression
from core.repositories.DataSetTypes import DataTypesEnum, NumericalDataTypesEnum, CategorialDataTypesEnum
from core.repositories.TaskTypes import MachineLearningTasks


class ModelTypesIdsEnum(Enum):
    xgboost = "xgbboost"
    knn = "knn"
    logit = "logit"


class ModelGroupsIds(Enum):
    ml = "ML_models"
    all = "Models"


class ModelMetaInfo(object):
    def __init__(self, input_types=None, output_types=None, task_types=None, can_be_initial=True,
                 can_be_secondary=True):
        self.input_types = input_types
        self.output_types = output_types
        self.task_type = task_types
        self.can_be_initial = can_be_initial
        self.can_be_secondary = can_be_secondary

    def compare_field_with_template(self, prop, template):
        if not isinstance(prop, list) and not (prop is None or not prop):
            prop = [prop]
        if not isinstance(template, list) and not (template is None or not template):
            template = [template]

        return prop is None or not prop or \
               template is None or not template or \
               any([prop_item in template for prop_item in prop])

    def suit_for_template(self, template):
        return self.compare_field_with_template(self.input_types, template.input_types) and \
               self.compare_field_with_template(self.output_types, template.output_types) and \
               self.compare_field_with_template(self.task_type, template.task_type) and \
               self.compare_field_with_template(self.can_be_initial, template.can_be_initial) and \
               self.compare_field_with_template(self.can_be_secondary, template.can_be_secondary)


class ModelsGroup(Node):
    def __init__(self, name, parent=None):
        super(Node, self).__init__()
        self.name = name
        self.parent = parent


class ModelType(Node):
    def __init__(self, name, meta_info, parent):
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
        root = ModelsGroup(ModelGroupsIds.all)

        ml = ModelsGroup(ModelGroupsIds.ml, parent=root)

        xgboost_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table, CategorialDataTypesEnum.table],
                                     output_types=[NumericalDataTypesEnum.vector, CategorialDataTypesEnum.vector],
                                     task_types=[MachineLearningTasks.classification, MachineLearningTasks.regression])

        xgboost = ModelType(ModelTypesIdsEnum.xgboost, xgboost_meta, parent=ml)

        knn_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table],
                                 output_types=[CategorialDataTypesEnum.vector],
                                 task_types=[MachineLearningTasks.classification])

        knn = ModelType(ModelTypesIdsEnum.knn, knn_meta, parent=ml)

        logit_meta = ModelMetaInfo(input_types=[NumericalDataTypesEnum.table, CategorialDataTypesEnum.table],
                                   output_types=[CategorialDataTypesEnum.vector],
                                   task_types=[MachineLearningTasks.classification])

        logit = ModelType(ModelTypesIdsEnum.logit, logit_meta, parent=ml)

        return root

    def __init__(self):
        self._tree = self._initialise_tree()

    def get_model_types_set_by_attributes(self, desired_ids, desired_metainfo=None):

        if desired_ids is None or not desired_ids:
            desired_ids = [ModelGroupsIds.all]

        if not isinstance(desired_ids, list):
            desired_ids = [desired_ids]

        results = findall(self._tree, filter_=lambda node:
        isinstance(node, ModelType) and
        any(node_from_path.name in desired_ids for node_from_path in node.path))

        if desired_metainfo is not None:
            results = [result for result in results
                       if (isinstance(result, ModelType) and result.meta_info.suit_for_template(desired_metainfo))]

        return [result.name for result in results]

    def obtain_model_implementation(self, model_type_id):
        return self.model_implementations[model_type_id]()

    def print_tree(self):
        for pre, fill, node in RenderTree(self._tree):
            print("%s%s" % (pre, node.name))