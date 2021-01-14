import numpy as np
import pandas as pd
import datetime
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.model_types_repository import (
    ModelTypesRepository
)
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum, \
    MetricsRepository, ClusteringMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def get_metric_function(task: Task):
    if task == Task(TaskTypesEnum.classification):
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
    elif task == Task(TaskTypesEnum.regression) or task == Task(TaskTypesEnum.ts_forecasting):
        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE_penalty),
    elif task == Task(TaskTypesEnum.clustering):
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


def only_light_model_types(available_model_types: list,
                           model_configuration: str):
    excluded_models_dict = {'light': ['mlp', 'svc'],
                            'default': []}
    excluded_models = excluded_models_dict[model_configuration]

    available_model_types = [model for model in available_model_types if model not in excluded_models]

    return available_model_types


def compose_fedot_model(train_data: InputData,
                        task: Task,
                        max_depth: int,
                        max_arity: int,
                        pop_size: int,
                        num_of_generations: int,
                        learning_time: int = 5,
                        model_types: list = None,
                        model_configuration: str = 'light',
                        ):
    # the choice of the metric for the chain quality assessment during composition
    metric_function = get_metric_function(task)

    learning_time = datetime.timedelta(minutes=learning_time)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    if model_types is not None:
        available_model_types = model_types

    only_light_model_types(available_model_types, model_configuration)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=max_arity,
        max_depth=max_depth, pop_size=pop_size, num_of_generations=num_of_generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=learning_time)

    # Create GP-based composer
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    gp_composer = builder.build()

    chain_gp_composed = gp_composer.compose_chain(data=train_data)

    chain_gp_composed.fit_from_scratch(input_data=train_data)

    return chain_gp_composed
