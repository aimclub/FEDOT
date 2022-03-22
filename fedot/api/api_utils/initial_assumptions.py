from random import choice
from typing import List, Union

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.log import Log

NOT_FITTED_ERR_MSG = 'Model not fitted yet'
UNSUITABLE_AVAILABLE_OPERATIONS_MSG = "Unable to construct an initial assumption from the passed " \
                                      "available operations, default initial assumption will be used"


class ApiInitialAssumptions:
    def get_initial_assumption(self,
                               data: Union[InputData, MultiModalData],
                               task: Task,
                               available_operations: List[str] = None,
                               logger: Log = None) -> List[Pipeline]:

        has_categorical_features = data_has_categorical_features(data)
        has_gaps = data_has_missing_values(data)

        if isinstance(data, MultiModalData):
            if available_operations:
                logger.message("Available operations are not taken into account when "
                               "forming the initial assumption for multi-modal data")
            initial_assumption = self.create_multidata_pipelines(task, data, has_categorical_features, has_gaps)
        elif isinstance(data, InputData):
            if available_operations:
                initial_assumption = \
                    self.create_unidata_pipelines_on_available_operations(task, data, has_categorical_features,
                                                                          has_gaps, available_operations,
                                                                          logger)
            else:
                initial_assumption = self.create_unidata_pipelines(task, has_categorical_features, has_gaps)
        else:
            raise NotImplementedError(f"Don't handle {type(data)}")
        return initial_assumption

    @staticmethod
    def _get_operations_for_the_task(task_type: TaskTypesEnum, data_type: DataTypesEnum, repo: str,
                                     available_operations: List[str]):
        """ Returns the intersection of the sets of passed available operations and
        operations that are suitable for solving the given problem """

        operations_for_the_task = \
            OperationTypesRepository(repo).suitable_operation(task_type=task_type,
                                                              data_type=data_type)[0]
        operations_to_choose_from = list(set(operations_for_the_task).intersection(available_operations))
        return operations_to_choose_from

    @staticmethod
    def _are_only_available_operations(pipeline: Pipeline, available_operations: List[str]):
        """ Checks if the pipeline contains only nodes with passed available operations """

        for node in pipeline.nodes:
            if node.operation.operation_type not in available_operations:
                return False
        return True

    def _create_unidata_pipeline_on_random_operation(self, task, data, pipeline, available_operations, logger):
        """ Creates pipeline from one model randomly selected from the pool of available operations.
        For time series problem, first node with 'lagged' operation, then the randomly selected model.
        If it is impossible to create a valid pipeline from the given available operations,
        returns the default one """

        if task.task_type == TaskTypesEnum.ts_forecasting:
            node_lagged = PrimaryNode('lagged')
            operations_to_choose_from = \
                self._get_operations_for_the_task(task_type=TaskTypesEnum.regression, data_type=data.data_type,
                                                  repo='model', available_operations=available_operations)
            if not operations_to_choose_from:
                logger.message(UNSUITABLE_AVAILABLE_OPERATIONS_MSG)
                return pipeline

            node_final = SecondaryNode(choice([operations_to_choose_from]), nodes_from=[node_lagged])
            return Pipeline(node_final)

        elif task.task_type == TaskTypesEnum.regression or \
                task.task_type == TaskTypesEnum.classification:
            operations_to_choose_from = \
                self._get_operations_for_the_task(task_type=task.task_type, data_type=data.data_type,
                                                  repo='model', available_operations=available_operations)
            if not operations_to_choose_from:
                logger.message(UNSUITABLE_AVAILABLE_OPERATIONS_MSG)
                return pipeline

            node = PrimaryNode(choice(operations_to_choose_from))
            return Pipeline(node)
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

    def create_unidata_pipelines_on_available_operations(self, task: Task, data: InputData,
                                                         has_categorical_features: bool, has_gaps: bool,
                                                         available_operations: List[str],
                                                         logger: Log) -> List[Pipeline]:
        """ Creates a pipeline for Uni-data using only available operations """

        pipelines = self.create_unidata_pipelines(task, has_categorical_features, has_gaps)
        correct_pipelines = []
        for pipeline in pipelines:
            if self._are_only_available_operations(pipeline, available_operations):
                correct_pipelines.append(pipeline)
            else:
                correct_pipeline = self._create_unidata_pipeline_on_random_operation(task, data,
                                                                                     pipeline, available_operations,
                                                                                     logger)
                correct_pipelines.append(correct_pipeline)
        return correct_pipelines

    def create_unidata_pipelines(self,
                                 task: Task,
                                 has_categorical_features: bool,
                                 has_gaps: bool) -> List[Pipeline]:
        # TODO refactor as builder
        node_preprocessed = preprocessing_builder(task.task_type, has_gaps, has_categorical_features)
        if task.task_type == TaskTypesEnum.ts_forecasting:
            pipelines = [create_glm_ridge_pipeline(node_preprocessed),
                         create_lagged_ridge_pipeline(node_preprocessed),
                         create_polyfit_ridge_pipeline(node_preprocessed),
                         create_ar_pipeline(node_preprocessed)]
        elif task.task_type == TaskTypesEnum.classification:
            if has_categorical_features:
                pipelines = [create_rf_classifier_pipeline(node_preprocessed),
                             create_logit_classifier_pipeline(node_preprocessed)]
            else:
                pipelines = [create_rf_classifier_pipeline(node_preprocessed),
                             create_logit_classifier_pipeline(node_preprocessed)]
        elif task.task_type == TaskTypesEnum.regression:
            if has_categorical_features:
                pipelines = [create_rfr_regression_pipeline(node_preprocessed),
                             create_ridge_regression_pipeline(node_preprocessed)]
            else:
                pipelines = [create_rfr_regression_pipeline(node_preprocessed),
                             create_ridge_regression_pipeline(node_preprocessed)]
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
    """ Pipeline for time series forecasting task """
    if node_preprocessed:
        node_lagged = SecondaryNode('lagged', nodes_from=[node_preprocessed])
    else:
        node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_final)


def create_glm_ridge_pipeline(node_preprocessed=None):
    """ Pipeline for time series forecasting task """
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
    """ Pipeline for time series forecasting task """
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
    """ Pipeline for time series forecasting task """
    if node_preprocessed:
        node_smoothing = SecondaryNode('smoothing', nodes_from=[node_preprocessed])
    else:
        node_smoothing = PrimaryNode('smoothing')
    node_final = SecondaryNode('ar', nodes_from=[node_smoothing])
    return Pipeline(node_final)


def create_rf_classifier_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('rf', nodes_from=[node_preprocessed]))


def create_logit_classifier_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('logit', nodes_from=[node_preprocessed]))


def create_rfr_regression_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('rfr', nodes_from=[node_preprocessed]))


def create_ridge_regression_pipeline(node_preprocessed):
    return Pipeline(SecondaryNode('ridge', nodes_from=[node_preprocessed]))
