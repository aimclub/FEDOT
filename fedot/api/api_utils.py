import datetime

import numpy as np
import pandas as pd

from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import (
    ModelTypesRepository
)
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ClusteringMetricsEnum, \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def get_metric_function(task: Task):
    if task.task_type == TaskTypesEnum.classification:
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    elif task.task_type == TaskTypesEnum.regression or task.task_type == TaskTypesEnum.ts_forecasting:
        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    elif task.task_type == TaskTypesEnum.clustering:
        metric_function = MetricsRepository().metric_by_id(ClusteringMetricsEnum.silhouette)
    else:
        raise ValueError('Incorrect type of ML task')
    return metric_function


def save_predict(predicted_data: OutputData):
    if len(predicted_data.predict.shape) >= 2:
        prediction = predicted_data.predict.tolist()
    else:
        prediction = predicted_data.predict
    return pd.DataFrame({'Index': predicted_data.idx,
                         'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        task_type: Task = Task(TaskTypesEnum.classification)):
    data_type = DataTypesEnum.table
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task_type, data_type=data_type)


def filter_models_by_preset(available_model_types: list,
                            model_configuration: str):
    excluded_models_dict = {'light': ['mlp', 'svc'],
                            'light_tun': ['mlp', 'svc']}

    if model_configuration in excluded_models_dict.keys():
        excluded_models = excluded_models_dict[model_configuration]
        available_model_types = [_ for _ in available_model_types if _ not in excluded_models]

    if model_configuration in ['ultra_light', 'ultra_light_tun']:
        included_models = ['dt', 'dtreg', 'logit', 'linear', 'lasso', 'ridge', 'knn']
        available_model_types = [_ for _ in available_model_types if _ in included_models]

    return available_model_types


def compose_fedot_model(train_data: InputData,
                        task: Task,
                        logger: Log,
                        max_depth: int,
                        max_arity: int,
                        pop_size: int,
                        num_of_generations: int,
                        learning_time: int = 5,
                        model_types: list = None,
                        preset: str = 'light_tun'
                        ):
    # the choice of the metric for the chain quality assessment during composition
    metric_function = get_metric_function(task)

    is_tuning = '_tun' in preset or preset == 'full'

    learning_time = datetime.timedelta(minutes=learning_time)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    if model_types is not None:
        available_model_types = model_types

    available_model_types = filter_models_by_preset(available_model_types, preset)

    logger.message(f'{preset} preset is used. Parameters tuning: {is_tuning}. '
                   f'Set of candidate models: {available_model_types}. Composing time limit: {learning_time}')

    # the choice and initialisation of the GP composer
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=max_arity,
        max_depth=max_depth, pop_size=pop_size, num_of_generations=num_of_generations,
        max_lead_time=learning_time)

    # Create GP-based composer
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function). \
        with_logger(logger)
    gp_composer = builder.build()

    logger.message('Model composition started')
    chain_gp_composed = gp_composer.compose_chain(data=train_data)
    chain_gp_composed.log = logger

    if is_tuning:
        logger.message('Hyperparameters tuning started')
        chain_gp_composed.fine_tune_primary_nodes(input_data=train_data)

    logger.message('Model composition finished')

    return chain_gp_composed
