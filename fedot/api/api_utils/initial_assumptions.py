from typing import List, Union

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class ApiInitialAssumptions:
    def get_initial_assumption(self,
                               data: Union[InputData, MultiModalData],
                               task: Task) -> List[Pipeline]:

        has_categorical_features = data_has_categorical_features(data)
        has_gaps = data_has_missing_values(data)

        if isinstance(data, MultiModalData):
            initial_assumption = self.create_multidata_pipelines(task, data, has_categorical_features, has_gaps)
        elif isinstance(data, InputData):
            initial_assumption = self.create_unidata_pipelines(task, has_categorical_features, has_gaps)
        else:
            raise NotImplementedError(f"Don't handle {type(data)}")
        return initial_assumption

    def create_unidata_pipelines(self,
                                 task: Task,
                                 has_categorical_features: bool,
                                 has_gaps: bool) -> List[Pipeline]:
        # TODO refactor as builder
        node_prepocessed = preprocessing_builder(task.task_type, has_gaps, has_categorical_features)
        if task.task_type == TaskTypesEnum.ts_forecasting:
            pipelines = [create_glm_ridge_pipeline(node_prepocessed),
                         create_lagged_ridge_pipeline(node_prepocessed),
                         create_polyfit_ridge_pipeline(node_prepocessed),
                         create_ar_pipeline(node_prepocessed)]
        elif task.task_type == TaskTypesEnum.classification:
            if has_categorical_features:
                pipelines = [create_xgboost_classifier_pipeline(node_prepocessed)]
            else:
                pipelines = [create_xgboost_classifier_pipeline(node_prepocessed)]
        elif task.task_type == TaskTypesEnum.regression:
            if has_categorical_features:
                pipelines = [create_xgboost_regression_pipeline(node_prepocessed)]
            else:
                pipelines = [create_xgboost_regression_pipeline(node_prepocessed)]
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")
        return pipelines

    def create_multidata_pipelines(self, task: Task, data: MultiModalData,
                                   has_categorical_features: bool,
                                   has_gaps: bool) -> List[Pipeline]:
        if task.task_type == TaskTypesEnum.ts_forecasting:
            node_final = SecondaryNode('ridge', nodes_from=[])
            for data_source_name, values in data.items():
                if data_source_name.startswith('data_source_ts'):
                    node_primary = PrimaryNode(data_source_name)
                    node_lagged = SecondaryNode('lagged', [node_primary])
                    node_last = SecondaryNode('ridge', [node_lagged])
                    node_final.nodes_from.append(node_last)
        elif task.task_type == TaskTypesEnum.classification:
            node_final = SecondaryNode('xgboost', nodes_from=[])
            node_final.nodes_from = self.create_first_multimodal_nodes(data, has_categorical_features, has_gaps)
        elif task.task_type == TaskTypesEnum.regression:
            node_final = SecondaryNode('xgbreg', nodes_from=[])
            node_final.nodes_from = self.create_first_multimodal_nodes(data, has_categorical_features, has_gaps)
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

        return [Pipeline(node_final)]

    def create_first_multimodal_nodes(self, data: MultiModalData,
                                      has_categorical: bool, has_gaps: bool) -> List[Pipeline]:
        nodes_from = []

        for data_source_name, values in data.items():
            node_primary = PrimaryNode(data_source_name)
            node_imputation = SecondaryNode('simple_imputation', [node_primary])
            if has_gaps:
                if data_source_name.startswith('data_source_table') and has_categorical:
                    node_encoder = SecondaryNode('one_hot_encoding', [node_imputation])
                    node_preprocessing = SecondaryNode('scaling', [node_encoder])
                else:
                    node_preprocessing = SecondaryNode('scaling', [node_imputation])
            else:
                if data_source_name.startswith('data_source_table') and has_categorical:
                    node_encoder = SecondaryNode('one_hot_encoding', [node_primary])
                    node_preprocessing = SecondaryNode('scaling', [node_encoder])
                else:
                    node_preprocessing = SecondaryNode('scaling', [node_primary])
            node_last = SecondaryNode('ridge', [node_preprocessing])
            nodes_from.append(Pipeline(node_last))

        return nodes_from


def preprocessing_builder(task_type: TaskTypesEnum, has_gaps: bool = False, has_categorical_features: bool = False):
    """
    Function that accepts special info about data and create preprocessing part of pipeline

    :param task_type: type of task
    :param has_gaps: flag is showed is there are gaps in the data
    :param has_categorical_features: flag is showed is there are categorical_features
    :return: node_preprocessing: last node of preprocessing
    """
    node_imputation = PrimaryNode('simple_imputation')
    if task_type == TaskTypesEnum.ts_forecasting:
        if has_gaps:
            return node_imputation
    else:
        if has_gaps:
            if has_categorical_features:
                node_encoder = SecondaryNode('one_hot_encoding', nodes_from=[node_imputation])
                node_preprocessing = SecondaryNode('scaling', [node_encoder])
            else:
                node_preprocessing = SecondaryNode('scaling', nodes_from=[node_imputation])
        else:
            if has_categorical_features:
                node_encoder = PrimaryNode('one_hot_encoding')
                node_preprocessing = SecondaryNode('scaling', [node_encoder])
            else:
                node_preprocessing = PrimaryNode('scaling')
        return node_preprocessing


def create_lagged_ridge_pipeline(node_preprocessed=None):
    if node_preprocessed:
        node_lagged = SecondaryNode('lagged', nodes_from=[node_preprocessed])
    else:
        node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_final)


def create_glm_ridge_pipeline(node_preprocessed=None):
    if node_preprocessed:
        node_glm = SecondaryNode('glm', nodes_from=[node_preprocessed])
        node_lagged = SecondaryNode('lagged', nodes_from=[node_preprocessed])
    else:
        node_glm = PrimaryNode('glm')
        node_lagged = PrimaryNode('lagged')

    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge, node_glm])
    return Pipeline(node_final)


def create_polyfit_ridge_pipeline(node_preprocessed=None):
    if node_preprocessed:
        node_polyfit = SecondaryNode('polyfit', nodes_from=[node_preprocessed])
        node_lagged = SecondaryNode('lagged', nodes_from=[node_preprocessed])
    else:
        node_polyfit = PrimaryNode('polyfit')
        node_lagged = PrimaryNode('lagged')

    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge, node_polyfit])
    return Pipeline(node_final)


def create_ar_pipeline(node_preprocessed=None):
    if node_preprocessed:
        node_smoothing = SecondaryNode('smoothing', nodes_from=[node_preprocessed])
    else:
        node_smoothing = PrimaryNode('smoothing')
    node_final = SecondaryNode('ar', nodes_from=[node_smoothing])
    return Pipeline(node_final)


def create_xgboost_classifier_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('xgboost', nodes_from=[node_preprocessed]))


def create_xgboost_regression_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('xgbreg', nodes_from=[node_preprocessed]))
