import datetime
import os
import random
from pickle import dump, load
import numpy as np
from benchmark.benchmark_utils import get_models_hyperparameters
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def save_fedot_model(chain, path_to_file):
    with open(f'{path_to_file}.pkl', 'wb') as pickle_file:
        dump(chain, pickle_file)
    ComposerVisualiser.visualise(chain, f'{path_to_file}.png')


def load_fedot_model(path_to_file):
    if os.path.exists(path_to_file):
        with open(f'{path_to_file}.pkl', 'rb') as pickle_file:
            return load(pickle_file)
    else:
        return None


def run_fedot_for_problem(train_file_path, test_file_path, case_name, task_type: TaskTypesEnum):
    task = Task(task_type)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    models_hyperparameters = get_models_hyperparameters()['FEDOT']
    cur_lead_time = models_hyperparameters['MAX_RUNTIME_MINS']

    saved_model_name = f'fedot_{case_name}_{task_type}_{cur_lead_time}'
    loaded_model = load_fedot_model(saved_model_name)

    if not loaded_model:
        generations = models_hyperparameters['GENERATIONS']
        population_size = models_hyperparameters['POPULATION_SIZE']

        # the search of the models provided by the framework that can be used as nodes in a chain'
        models_repo = ModelTypesRepository()
        available_model_types, _ = models_repo.search_models(
            desired_metainfo=ModelMetaInfoTemplate(input_types=[DataTypesEnum.table],
                                                   task_type=task.task_type,
                                                   can_be_secondary=True))

        if task_type == TaskTypesEnum.classification:
            metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
        elif task_type == TaskTypesEnum.regression:
            metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
        else:
            raise NotImplementedError()

        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=3,
            max_depth=3, pop_size=population_size, num_of_generations=generations,
            crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

        # Create GP-based composer
        composer = GPComposer()

        # the optimal chain generation by composition - the most time-consuming task
        chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                    initial_chain=None,
                                                    composer_requirements=composer_requirements,
                                                    metrics=metric_function, is_visualise=False)
        chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose, iterations=50)
        chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)
        save_fedot_model(chain_evo_composed, f'fedot_{case_name}_{task_type}_{cur_lead_time}')
    else:
        chain_evo_composed = loaded_model

    evo_predicted = chain_evo_composed.predict(dataset_to_validate)

    return evo_predicted.predict, dataset_to_validate.target