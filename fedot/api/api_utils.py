import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_squared_error,
                             r2_score, roc_auc_score, mean_absolute_error)

from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task

from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, \
    ClusteringMetricsEnum, RegressionMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask, TunerMetricByTask
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.operation_types_repository import get_ts_operations


composer_metrics_mapping = {
    'acc': ClassificationMetricsEnum.accuracy,
    'roc_auc': ClassificationMetricsEnum.ROCAUC,
    'f1': ClassificationMetricsEnum.f1,
    'logloss': ClassificationMetricsEnum.logloss,
    'mae': RegressionMetricsEnum.MAE,
    'mse': RegressionMetricsEnum.MSE,
    'msle': RegressionMetricsEnum.MSLE,
    'r2': RegressionMetricsEnum.R2,
    'rmse': RegressionMetricsEnum.RMSE,
    'silhouette': ClusteringMetricsEnum.silhouette
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


def autodetect_data_type(task: Task) -> DataTypesEnum:
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
    data_type = autodetect_data_type(task)
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)


def filter_operations_by_preset(task, preset: str):
    """ Function filter operations by preset, remove "heavy" operations and save
    appropriate ones
    """
    excluded_models_dict = {'light': ['mlp', 'svc', 'arima', 'exog', 'text_clean'],
                            'light_tun': ['mlp', 'svc', 'arima', 'exog', 'text_clean']}

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


def compose_fedot_model(train_data: InputData,
                        task: Task,
                        logger: Log,
                        max_depth: int,
                        max_arity: int,
                        pop_size: int,
                        num_of_generations: int,
                        learning_time: int = 5,
                        available_operations: list = None,
                        composer_metric=None,
                        with_tuning=False,
                        tuner_metric=None
                        ):
    """ Function for composing FEDOT chain model """
    # the choice of the metric for the chain quality assessment during composition
    if composer_metric is None:
        metric_function = MetricByTask(task.task_type).metric_cls.get_value
    else:
        metric_id = composer_metrics_mapping.get(composer_metric, None)
        if metric_id is None:
            raise ValueError(f'Incorrect composer metric {composer_metric}')
        metric_function = MetricsRepository().metric_by_id(metric_id)

    learning_time = datetime.timedelta(minutes=learning_time)

    if available_operations is None:
        available_operations = get_operations_for_task(task, mode='models')

    logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                   f'Set of candidate models: {available_operations}. Composing time limit: {learning_time}')

    primary_operations, secondary_operations = divide_operations(available_operations,
                                                                 task)
    # the choice and initialisation of the GP composer
    composer_requirements = GPComposerRequirements(primary=primary_operations,
                                                   secondary=secondary_operations,
                                                   max_arity=max_arity,
                                                   max_depth=max_depth,
                                                   pop_size=pop_size,
                                                   num_of_generations=num_of_generations,
                                                   max_lead_time=learning_time,
                                                   allow_single_operations=False)

    # Create GP-based composer
    builder = get_gp_composer_builder(task=task,
                                      metric_function=metric_function,
                                      composer_requirements=composer_requirements,
                                      logger=logger)
    gp_composer = builder.build()

    logger.message('Model composition started')
    chain_gp_composed = gp_composer.compose_chain(data=train_data)
    chain_gp_composed.log = logger

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

        # Tune all nodes in the chain
        chain_gp_composed.fine_tune_all_nodes(loss_function=tuner_loss,
                                              loss_params=loss_params,
                                              input_data=train_data,
                                              iterations=20)

    logger.message('Model composition finished')

    return chain_gp_composed


def get_gp_composer_builder(task: Task, metric_function, composer_requirements,
                            logger):
    """ Return GPComposerBuilder with parameters and if it is necessary
    init_chain in it """
    if task.task_type == TaskTypesEnum.ts_forecasting:
        # Create init chain
        node_lagged = PrimaryNode('lagged')
        node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
        init_chain = Chain(node_final)

        builder = GPComposerBuilder(task=task). \
            with_requirements(composer_requirements). \
            with_metrics(metric_function).with_initial_chain(init_chain)
    else:
        builder = GPComposerBuilder(task).with_requirements(composer_requirements). \
            with_metrics(metric_function).with_logger(logger)

    return builder


def divide_operations(available_operations, task):
    """ Function divide operations for primary and secondary """

    if task.task_type == TaskTypesEnum.ts_forecasting:
        ts_data_operations = get_ts_operations(mode='data_operations',
                                               tags=["ts_specific"])
        # Remove exog data operation from the list
        ts_data_operations.remove('exog')

        primary_operations = ts_data_operations
        secondary_operations = available_operations
    else:
        primary_operations = available_operations
        secondary_operations = available_operations
    return primary_operations, secondary_operations
