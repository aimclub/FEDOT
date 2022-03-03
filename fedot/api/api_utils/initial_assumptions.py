from random import choice
from typing import List, Union

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.composer.gp_composer.specific_operators import filter_pipeline_with_available_operations

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class ApiInitialAssumptions:
    def get_initial_assumption(self,
                               data: Union[InputData, MultiModalData],
                               task: Task,
                               available_operations: List[str] = None) -> List[Pipeline]:

        has_categorical_features = data_has_categorical_features(data)
        has_gaps = data_has_missing_values(data)

        if isinstance(data, MultiModalData):
            if available_operations:
                initial_assumption = \
                    self.create_multidata_pipelines_on_available_operations(task, data, has_categorical_features,
                                                                            has_gaps, available_operations)
            else:
                initial_assumption = self.create_multidata_pipelines(task, data, has_categorical_features, has_gaps)
        elif isinstance(data, InputData):
            if available_operations:
                initial_assumption = \
                    self.create_unidata_pipelines_on_available_operations(task, data, has_categorical_features,
                                                                          has_gaps, available_operations)
            else:
                initial_assumption = self.create_unidata_pipelines(task, has_categorical_features, has_gaps)
        else:
            raise NotImplementedError(f"Don't handle {type(data)}")
        return initial_assumption

    @staticmethod
    def _get_non_repeating_operations(task: Task, data: Union[InputData, MultiModalData],
                                      available_operations: List[str], used_operations: List[str]):
        """ Returns operations that can be used to further form the pipeline and which are not yet in it

        :param task: task
        :param data: data
        :param available_operations: operations that are set to form a pipeline
        :param used_operations: operations that are already used in the pipeline
        """

        operations = OperationTypesRepository('all').suitable_operation(task_type=task.task_type,
                                                                        data_type=data.data_type)[0]
        operations_to_choose_from = [operation for operation in operations if operation in available_operations]
        if not operations_to_choose_from:
            raise ValueError(f"The specified avaialable operations: {available_operations} are "
                             f"not suitable for solving {task.task_type} task")

        non_repeating_operations = [operation for operation in operations_to_choose_from
                                    if operation not in used_operations]
        return non_repeating_operations

    def create_unidata_pipelines_on_available_operations(self, task: Task, data: InputData,
                                                         has_categorical_features: bool, has_gaps: bool,
                                                         available_operations: List[str]) -> List[Pipeline]:
        """ Creates a pipeline for Uni-data using only available operations """

        node_prepocessed = preprocessing_builder(task.task_type, has_gaps, has_categorical_features)
        preprocessing_operations = [node.operation.operation_type
                                    for node in node_prepocessed.ordered_subnodes_hierarchy()]

        non_repeating_operations = self._get_non_repeating_operations(task, data, available_operations,
                                                                      preprocessing_operations)
        if not non_repeating_operations:
            return [Pipeline(node_prepocessed)]

        node_operation = choice(non_repeating_operations)
        secondary_node = SecondaryNode(node_operation, nodes_from=[node_prepocessed])
        pipeline = Pipeline(secondary_node)

        filter_pipeline_with_available_operations(pipeline=pipeline, available_operations=available_operations)
        return [pipeline]

    def create_multidata_pipelines_on_available_operations(self, task: Task, data: MultiModalData,
                                                           has_categorical_features: bool,
                                                           has_gaps: bool,
                                                           available_operations: List[str]) -> List[Pipeline]:
        """ Creates a pipeline for Multi-data using only available operations """

        if task.task_type == TaskTypesEnum.ts_forecasting:
            node = PrimaryNode(choice(available_operations))
            pipeline = Pipeline(node)
        elif task.task_type == TaskTypesEnum.classification or \
                task.task_type == TaskTypesEnum.regression:
            first_nodes_pipe = self.create_first_multimodal_nodes(data, has_categorical_features, has_gaps)[0]
            first_operations = [node.operation.operation_type for node in first_nodes_pipe.nodes]

            non_repeating_operations = self._get_non_repeating_operations(task, data,
                                                                          available_operations, first_operations)
            if not non_repeating_operations:
                return [first_nodes_pipe]

            node_operation = choice(non_repeating_operations)
            secondary_node = SecondaryNode(node_operation, nodes_from=[first_nodes_pipe.root_node])
            pipeline = Pipeline(secondary_node)

            filter_pipeline_with_available_operations(pipeline=pipeline, available_operations=available_operations)
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")
        return [pipeline]

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
                pipelines = [create_rf_classifier_pipeline(node_prepocessed)]
            else:
                pipelines = [create_rf_classifier_pipeline(node_prepocessed)]
        elif task.task_type == TaskTypesEnum.regression:
            if has_categorical_features:
                pipelines = [create_rfr_regression_pipeline(node_prepocessed)]
            else:
                pipelines = [create_rfr_regression_pipeline(node_prepocessed)]
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
            node_final = SecondaryNode('rf', nodes_from=[])
            node_final.nodes_from = self.create_first_multimodal_nodes(data, has_categorical_features, has_gaps)
        elif task.task_type == TaskTypesEnum.regression:
            node_final = SecondaryNode('rfr', nodes_from=[])
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


def create_rf_classifier_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('rf', nodes_from=[node_preprocessed]))


def create_rfr_regression_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('rfr', nodes_from=[node_preprocessed]))
