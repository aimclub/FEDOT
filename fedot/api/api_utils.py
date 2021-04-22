import datetime
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from fedot.core.composer.gp_composer.gp_composer import (GPComposerBuilder, GPComposerRequirements,
                                                         GPGraphOptimiserParameters)
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task, get_ts_operations
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


def filter_operations_by_preset(task, preset: str):
    """ Function filter operations by preset, remove "heavy" operations and save
    appropriate ones
    """
    excluded_models_dict = {'light': ['mlp', 'svc', 'arima', 'exog_ts_data_source', 'text_clean', 'catboost',
                                      'lgbm', 'lgbmreg', 'catboostreg'],
                            'light_tun': ['mlp', 'svc', 'arima', 'exog_ts_data_source', 'text_clean', 'catboost',
                                          'catboostreg', 'lgbm', 'lgbmreg']}

    # Get data operations and models
    available_operations = get_operations_for_task(task, mode='all')
    available_data_operation = get_operations_for_task(task, mode='data_operations')

    # Exclude "heavy" operations if necessary
    if preset in excluded_models_dict.keys():
        excluded_operations = excluded_models_dict[preset]
        available_operations = [_ for _ in available_operations if _ not in excluded_operations]

    # Save only "light" operations
    if preset in ['ultra_light', 'ultra_light_tun']:
        light_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn', 'ar']
        included_operations = light_models + available_data_operation
        available_operations = [_ for _ in available_operations if _ in included_operations]

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
                        learning_time: float = 5,
                        with_tuning=False,
                        tuner_metric=None,
                        cv_folds: Optional[int] = None,
                        initial_chain=None
                        ):
    """ Function for composing FEDOT pipeline """

    metric_function = _obtain_metric(task, composer_metric)

    if available_operations is None:
        available_operations = get_operations_for_task(task, mode='models')

    logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                   f'Set of candidate models: {available_operations}. Composing time limit: {learning_time} min')

    primary_operations, secondary_operations = _divide_operations(available_operations,
                                                                  task)

    learning_time_for_composing = learning_time / 2 if with_tuning else learning_time
    # the choice and initialisation of the GP composer
    composer_requirements = \
        GPComposerRequirements(primary=primary_operations,
                               secondary=secondary_operations,
                               max_arity=max_arity,
                               max_depth=max_depth,
                               pop_size=pop_size,
                               num_of_generations=num_of_generations,
                               timeout=datetime.timedelta(minutes=learning_time_for_composing),
                               cv_folds=cv_folds)

    optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
                                                      mutation_types=[parameter_change_mutation,
                                                                      MutationTypesEnum.simple,
                                                                      MutationTypesEnum.reduce,
                                                                      MutationTypesEnum.growth,
                                                                      MutationTypesEnum.local_growth],
                                                      crossover_types=[CrossoverTypesEnum.one_point,
                                                                       CrossoverTypesEnum.subtree])

    # Create GP-based composer
    builder = _get_gp_composer_builder(task=task,
                                       metric_function=metric_function,
                                       composer_requirements=composer_requirements,
                                       optimizer_parameters=optimizer_parameters,
                                       data=train_data,
                                       initial_chain=initial_chain,
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
            logger.message('Default loss function was set')
            # Default metric for tuner
            tune_metrics = TunerMetricByTask(task.task_type)
            tuner_loss, loss_params = tune_metrics.get_metric_and_params(train_data)
        else:
            # Get metric and parameters by name
            tuner_loss, loss_params = tuner_metric_by_name(metric_name=tuner_metric,
                                                           train_data=train_data,
                                                           task=task)

        iterations = 20 if learning_time is None else 1000
        learning_time_for_tuning = learning_time / 2

        # Tune all nodes in the pipeline
        pipeline_for_return.fine_tune_all_nodes(loss_function=tuner_loss,
                                                loss_params=loss_params,
                                                input_data=train_data,
                                                iterations=iterations, timeout=learning_time_for_tuning)

    logger.message('Model composition finished')

    history = gp_composer.optimiser.history

    return pipeline_for_return, best_candidates, history


def _obtain_initial_assumption(task: Task, data) -> Pipeline:
    node_final = None
    if task.task_type == TaskTypesEnum.ts_forecasting:
        # Create init pipeline
        if isinstance(data, MultiModalData):
            node_final = SecondaryNode('ridge', nodes_from=[])
            for data_source_name in data.keys():
                last_node_for_sub_pipeline = \
                    SecondaryNode('ridge', [SecondaryNode('lagged', [PrimaryNode(data_source_name)])])
                node_final.nodes_from.append(last_node_for_sub_pipeline)
        else:
            node_final = SecondaryNode('ridge', nodes_from=[PrimaryNode('lagged')])
    elif task.task_type == TaskTypesEnum.classification:
        node_lagged = PrimaryNode('scaling')
        node_final = SecondaryNode('xgboost', nodes_from=[node_lagged])
    elif task.task_type == TaskTypesEnum.regression:
        node_lagged = PrimaryNode('scaling')
        node_final = SecondaryNode('ridge', nodes_from=[node_lagged])

    init_pipeline = Pipeline(node_final)
    return init_pipeline


def _get_gp_composer_builder(task: Task, metric_function,
                             composer_requirements: GPComposerRequirements,
                             optimizer_parameters: GPGraphOptimiserParameters,
                             data: Union[InputData, MultiModalData],
                             initial_chain: Pipeline,
                             logger: Log):
    """ Return GPComposerBuilder with parameters and if it is necessary
    init_pipeline in it """

    builder = GPComposerBuilder(task=task). \
        with_requirements(composer_requirements). \
        with_optimiser_parameters(optimizer_parameters). \
        with_metrics(metric_function).with_logger(logger)

    init_pipeline = _obtain_initial_assumption(task, data) if not initial_chain else initial_chain

    if init_pipeline is not None:
        builder = builder.with_initial_pipeline(init_pipeline)

    return builder


def _divide_operations(available_operations, task):
    """ Function divide operations for primary and secondary """

    if task.task_type == TaskTypesEnum.ts_forecasting:
        ts_data_operations = get_ts_operations(mode='data_operations',
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
