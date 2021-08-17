import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from fedot.core.composer.gp_composer.gp_composer import (GPComposerBuilder, GPComposerRequirements,
                                                         GPGraphOptimiserParameters)
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.data.data import InputData, OutputData, data_has_categorical_features
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import single_add_mutation, single_change_mutation, \
    single_drop_mutation, single_edge_mutation
from fedot.core.pipelines.node import Node, PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operations_for_task
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, MetricsRepository,
                                                              RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask

composer_metrics_mapping = {
    'acc': ClassificationMetricsEnum.accuracy,
    'roc_auc': ClassificationMetricsEnum.ROCAUC,
    'f1': ClassificationMetricsEnum.f1,
    'logloss': ClassificationMetricsEnum.logloss,
    'mae': RegressionMetricsEnum.MAE,
    'mse': RegressionMetricsEnum.MSE,
    'msle': RegressionMetricsEnum.MSLE,
    'mape': RegressionMetricsEnum.MAPE,
    'r2': RegressionMetricsEnum.R2,
    'rmse': RegressionMetricsEnum.RMSE,
    'rmse_pen': RegressionMetricsEnum.RMSE_penalty,
    'silhouette': ClusteringMetricsEnum.silhouette,
    'node_num': ComplexityMetricsEnum.node_num
}

tuner_metrics_mapping = {
    'acc': accuracy_score,
    'roc_auc': roc_auc_score,
    'f1': f1_score,
    'logloss': log_loss,
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score,
    'rmse': mean_squared_error,
}


def tuner_metric_by_name(metric_name: str, train_data: InputData, task: Task):
    """ Function allow to obtain metric for tuner by its name

    :param metric_name: name of metric
    :param train_data: InputData for train
    :param task: task to solve

    :return tuner_loss: loss function for tuner
    :return loss_params: parameters for tuner loss (can be None in some cases)
    """
    loss_params = None
    tuner_loss = tuner_metrics_mapping.get(metric_name)
    if tuner_loss is None:
        raise ValueError(f'Incorrect tuner metric {tuner_loss}')

    if metric_name == 'rmse':
        loss_params = {'squared': False}
    elif metric_name == 'roc_auc' and task == TaskTypesEnum.classification:
        amount_of_classes = len(np.unique(np.array(train_data.target)))
        if amount_of_classes == 2:
            # Binary classification
            loss_params = None
        else:
            # Metric for multiclass classification
            loss_params = {'multi_class': 'ovr'}
    return tuner_loss, loss_params


def _autodetect_data_type(task: Task) -> DataTypesEnum:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    else:
        return DataTypesEnum.table


def save_predict(predicted_data: OutputData):
    if len(predicted_data.predict.shape) >= 2:
        prediction = predicted_data.predict.tolist()
    else:
        prediction = predicted_data.predict
    return pd.DataFrame({'Index': predicted_data.idx,
                         'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        task: Task = Task(TaskTypesEnum.classification)):
    data_type = _autodetect_data_type(task)
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


def filter_operations_by_preset(task: Task, preset: str):
    """ Function filter operations by preset, remove "heavy" operations and save
    appropriate ones
    """
    excluded_models_dict = {'light': ['mlp', 'svc', 'arima', 'exog_ts_data_source', 'text_clean'],
                            'light_tun': ['mlp', 'svc', 'arima', 'exog_ts_data_source', 'text_clean']}

    # Get data operations and models
    available_operations = get_operations_for_task(task, mode='all')
    available_data_operation = get_operations_for_task(task, mode='data_operation')

    # Exclude "heavy" operations if necessary
    if preset in excluded_models_dict.keys():
        excluded_operations = excluded_models_dict[preset]
        available_operations = [_ for _ in available_operations if _ not in excluded_operations]

    # Save only "light" operations
    if preset in ['ultra_light', 'ultra_light_tun']:
        light_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn', 'ar']
        included_operations = light_models + available_data_operation
        available_operations = [_ for _ in available_operations if _ in included_operations]

    if preset == 'gpu':
        # OperationTypesRepository.assign_repo('model', 'gpu_models_repository.json')
        repository = OperationTypesRepository().assign_repo('model', 'gpu_models_repository.json')
        available_operations = repository.suitable_operation(task_type=task.task_type)
    return available_operations


def compose_fedot_model(train_data: [InputData, MultiModalData],
                        task: Task,
                        logger: Log,
                        max_depth: int,
                        max_arity: int,
                        pop_size: int,
                        num_of_generations: int,
                        available_operations: list = None,
                        composer_metric=None,
                        timeout: float = 5,
                        with_tuning=False,
                        tuner_metric=None,
                        cv_folds: Optional[int] = None,
                        validation_blocks: int = None,
                        initial_pipeline=None,
                        genetic_scheme: str = None
                        ):
    """ Function for composing FEDOT pipeline """

    metric_function = _obtain_metric(task, composer_metric)

    if available_operations is None:
        available_operations = get_operations_for_task(task, mode='model')

    logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                   f'Set of candidate models: {available_operations}. Composing time limit: {timeout} min')

    primary_operations, secondary_operations = _divide_operations(available_operations,
                                                                  task)

    timeout_for_composing = timeout / 2 if with_tuning else timeout
    # the choice and initialisation of the GP composer
    composer_requirements = \
        GPComposerRequirements(primary=primary_operations,
                               secondary=secondary_operations,
                               max_arity=max_arity,
                               max_depth=max_depth,
                               pop_size=pop_size,
                               num_of_generations=num_of_generations,
                               cv_folds=cv_folds,
                               validation_blocks=validation_blocks,
                               timeout=datetime.timedelta(minutes=timeout_for_composing))

    genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

    if genetic_scheme == 'steady_state':
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state

    optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=genetic_scheme_type,
                                                      mutation_types=[boosting_mutation, parameter_change_mutation,
                                                                      single_edge_mutation, single_change_mutation,
                                                                      single_drop_mutation,
                                                                      single_add_mutation],
                                                      crossover_types=[CrossoverTypesEnum.one_point,
                                                                       CrossoverTypesEnum.subtree])

    # Create GP-based composer
    builder = _get_gp_composer_builder(task=task,
                                       metric_function=metric_function,
                                       composer_requirements=composer_requirements,
                                       optimizer_parameters=optimizer_parameters,
                                       data=train_data,
                                       initial_pipeline=initial_pipeline,
                                       logger=logger)
    gp_composer = builder.build()

    logger.message('Pipeline composition started')
    pipeline_gp_composed = gp_composer.compose_pipeline(data=train_data)

    pipeline_for_return = pipeline_gp_composed

    if isinstance(pipeline_gp_composed, list):
        for pipeline in pipeline_gp_composed:
            pipeline.log = logger
        pipeline_for_return = pipeline_gp_composed[0]
        best_candidates = gp_composer.optimiser.archive
    else:
        best_candidates = [pipeline_gp_composed]
        pipeline_gp_composed.log = logger

    if with_tuning:
        logger.message('Hyperparameters tuning started')

        if tuner_metric is None:
            # Default metric for tuner
            tune_metrics = TunerMetricByTask(task.task_type)
            tuner_loss, loss_params = tune_metrics.get_metric_and_params(train_data)
            logger.message(f'Tuner metric is None, '
                           f'{tuner_loss.__name__} was set as default')
        else:
            # Get metric and parameters by name
            tuner_loss, loss_params = tuner_metric_by_name(metric_name=tuner_metric,
                                                           train_data=train_data,
                                                           task=task)

        iterations = 20 if timeout is None else 1000
        timeout_for_tuning = timeout / 2

        # Tune all nodes in the pipeline

        vb_number = composer_requirements.validation_blocks
        folds = composer_requirements.cv_folds
        if train_data.task.task_type != TaskTypesEnum.ts_forecasting:
            # TODO remove after implementation of CV for class/regr
            logger.warn('Cross-validation is not supported for tuning of ts-forecasting pipeline: '
                        'hold-out validation used instead')
            folds = None
        pipeline_for_return = pipeline_for_return.fine_tune_all_nodes(loss_function=tuner_loss,
                                                                      loss_params=loss_params,
                                                                      input_data=train_data,
                                                                      iterations=iterations,
                                                                      timeout=timeout_for_tuning,
                                                                      cv_folds=folds,
                                                                      validation_blocks=vb_number)

    logger.message('Model composition finished')

    history = gp_composer.optimiser.history

    return pipeline_for_return, best_candidates, history


def _create_unidata_pipeline(task: Task, has_categorical_features: bool) -> Node:
    node_imputation = PrimaryNode('simple_imputation')
    if task.task_type == TaskTypesEnum.ts_forecasting:
        node_lagged = SecondaryNode('lagged', [node_imputation])
        node_final = SecondaryNode('ridge', [node_lagged])
    else:
        if has_categorical_features:
            node_encoder = SecondaryNode('one_hot_encoding', [node_imputation])
            node_preprocessing = SecondaryNode('scaling', [node_encoder])
        else:
            node_preprocessing = SecondaryNode('scaling', [node_imputation])

        if task.task_type == TaskTypesEnum.classification:
            node_final = SecondaryNode('xgboost', nodes_from=[node_preprocessing])
        elif task.task_type == TaskTypesEnum.regression:
            node_final = SecondaryNode('xgbreg', nodes_from=[node_preprocessing])
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

    return node_final


def _create_multidata_pipeline(task: Task, data: MultiModalData, has_categorical_features: bool) -> Node:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        node_final = SecondaryNode('ridge', nodes_from=[])
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_ts'):
                node_primary = PrimaryNode(data_source_name)
                node_imputation = SecondaryNode('simple_imputation', [node_primary])
                node_lagged = SecondaryNode('lagged', [node_imputation])
                node_last = SecondaryNode('ridge', [node_lagged])
                node_final.nodes_from.append(node_last)
    elif task.task_type == TaskTypesEnum.classification:
        node_final = SecondaryNode('xgboost', nodes_from=[])
        node_final.nodes_from = _create_first_multimodal_nodes(data, has_categorical_features)
    elif task.task_type == TaskTypesEnum.regression:
        node_final = SecondaryNode('xgbreg', nodes_from=[])
        node_final.nodes_from = _create_first_multimodal_nodes(data, has_categorical_features)
    else:
        raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

    return node_final


def _create_first_multimodal_nodes(data: MultiModalData, has_categorical: bool) -> List[SecondaryNode]:
    nodes_from = []

    for data_source_name, values in data.items():
        node_primary = PrimaryNode(data_source_name)
        node_imputation = SecondaryNode('simple_imputation', [node_primary])
        if data_source_name.startswith('data_source_table') and has_categorical:
            node_encoder = SecondaryNode('one_hot_encoding', [node_imputation])
            node_preprocessing = SecondaryNode('scaling', [node_encoder])
        else:
            node_preprocessing = SecondaryNode('scaling', [node_imputation])
        node_last = SecondaryNode('ridge', [node_preprocessing])
        nodes_from.append(node_last)

    return nodes_from


def _obtain_initial_assumption(task: Task, data: Union[InputData, MultiModalData]) -> Pipeline:
    has_categorical_features = data_has_categorical_features(data)

    if isinstance(data, MultiModalData):
        node_final = _create_multidata_pipeline(task, data, has_categorical_features)
    elif isinstance(data, InputData):
        node_final = _create_unidata_pipeline(task, has_categorical_features)
    else:
        raise NotImplementedError(f"Don't handle {type(data)}")

    init_pipeline = Pipeline(node_final)
    return init_pipeline


def _get_gp_composer_builder(task: Task, metric_function,
                             composer_requirements: GPComposerRequirements,
                             optimizer_parameters: GPGraphOptimiserParameters,
                             data: Union[InputData, MultiModalData],
                             initial_pipeline: Pipeline,
                             logger: Log):
    """ Return GPComposerBuilder with parameters and if it is necessary
    init_pipeline in it """

    builder = GPComposerBuilder(task=task). \
        with_requirements(composer_requirements). \
        with_optimiser_parameters(optimizer_parameters). \
        with_metrics(metric_function).with_logger(logger)

    init_pipeline = _obtain_initial_assumption(task, data) if not initial_pipeline else initial_pipeline

    if init_pipeline is not None:
        builder = builder.with_initial_pipeline(init_pipeline)

    return builder


def _divide_operations(available_operations, task):
    """ Function divide operations for primary and secondary """

    if task.task_type == TaskTypesEnum.ts_forecasting:
        ts_data_operations = get_operations_for_task(task=task,
                                                     mode='data_operation',
                                                     tags=["ts_specific"])
        # Remove exog data operation from the list
        ts_data_operations.remove('exog_ts_data_source')

        primary_operations = ts_data_operations
        secondary_operations = available_operations
    else:
        primary_operations = available_operations
        secondary_operations = available_operations
    return primary_operations, secondary_operations


def _obtain_metric(task: Task, composer_metric: Union[str, Callable]):
    # the choice of the metric for the pipeline quality assessment during composition
    if composer_metric is None:
        composer_metric = MetricByTask(task.task_type).metric_cls.get_value

    if isinstance(composer_metric, str) or isinstance(composer_metric, Callable):
        composer_metric = [composer_metric]

    metric_function = []
    for specific_metric in composer_metric:
        if isinstance(specific_metric, Callable):
            specific_metric_function = specific_metric
        else:
            metric_id = composer_metrics_mapping.get(specific_metric, None)
            if metric_id is None:
                raise ValueError(f'Incorrect metric {specific_metric}')
            specific_metric_function = MetricsRepository().metric_by_id(metric_id)
        metric_function.append(specific_metric_function)
    return metric_function
