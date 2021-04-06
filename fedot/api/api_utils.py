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
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, MetricsRepository,
                                                              RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.define_metric_by_task import MetricByTask

metrics_mapping = {
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
    'silhouette': ClusteringMetricsEnum.silhouette,
    'node_num': ComplexityMetricsEnum.node_num
}


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


def filter_models_by_preset(task, preset: str):
    excluded_models_dict = {'light': ['mlp', 'svc'],
                            'light_tun': ['mlp', 'svc']}

    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    if preset in excluded_models_dict.keys():
        excluded_models = excluded_models_dict[preset]
        available_model_types = [_ for _ in available_model_types if _ not in excluded_models]

    if preset in ['ultra_light', 'ultra_light_tun']:
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
                        learning_time: float = 5,
                        available_model_types: list = None,
                        with_tuning=False,
                        metric=None
                        ):
    # the choice of the metric for the chain quality assessment during composition
    if metric is None:
        metric_function = MetricByTask(task.task_type).metric_cls.get_value
    else:
        if isinstance(metric, str):
            metric = [metric]

        metric_function = []
        for specific_metric in metric:
            metric_id = metrics_mapping.get(specific_metric, None)
            if metric_id is None:
                raise ValueError(f'Incorrect metric {specific_metric}')
            specific_metric_function = MetricsRepository().metric_by_id(metric_id)
            metric_function.append(specific_metric_function)

    learning_time = datetime.timedelta(minutes=learning_time)

    if available_model_types is None:
        available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    logger.message(f'Composition started. Parameters tuning: {with_tuning}. '
                   f'Set of candidate models: {available_model_types}. Composing time limit: {learning_time}')

    # the choice and initialisation of the GP composer
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=max_arity,
        max_depth=max_depth, pop_size=pop_size, num_of_generations=num_of_generations,
        max_lead_time=learning_time)

    # Create GP-based composer
    builder = GPComposerBuilder(task).with_requirements(composer_requirements). \
        with_metrics(metric_function).with_logger(logger)
    gp_composer = builder.build()

    logger.message('Model composition started')
    chain_gp_composed = gp_composer.compose_chain(data=train_data)

    chain_for_tune = chain_gp_composed
    chain_for_return = chain_gp_composed

    if isinstance(chain_gp_composed, list):
        for chain in chain_gp_composed:
            chain.log = logger
        chain_for_tune = chain_gp_composed[0]
        chain_for_return = gp_composer.optimiser.archive

    if with_tuning:
        logger.message('Hyperparameters tuning started')
        chain_for_tune.fine_tune_primary_nodes(input_data=train_data)

    logger.message('Model composition finished')

    return chain_for_return
