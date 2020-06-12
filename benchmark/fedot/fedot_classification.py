import datetime
import random

from benchmark.benchmark_utils import get_models_hyperparameters
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.model import *
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def run_classification_problem(train_file_path, test_file_path, vis_flag: bool = False):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    models_hyperparameters = get_models_hyperparameters()['FEDOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']
    cur_lead_time = models_hyperparameters['MAX_RUNTIME_MINS']

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository(). \
        suitable_model(task_type=task.task_type)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=population_size, generations=generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose, iterations=50)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    # the choice and initialisation of the dummy_composer
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)

    chain_static = dummy_composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_static.fit(input_data=dataset_to_compose, verbose=True)
    # the single-model variant of optimal chain
    single_composer_requirements = ComposerRequirements(primary=['mlp'],
                                                        secondary=[])
    chain_single = DummyComposer(DummyChainTypeEnum.flat).compose_chain(
        data=dataset_to_compose,
        initial_chain=None,
        composer_requirements=single_composer_requirements,
        metrics=metric_function)
    chain_single.fit(input_data=dataset_to_compose, verbose=True)

    if vis_flag:
        ComposerVisualiser.visualise(chain_static)
        ComposerVisualiser.visualise(chain_evo_composed)

    static_predicted = chain_static.predict(dataset_to_validate)
    single_predicted = chain_single.predict(dataset_to_validate)
    evo_predicted = chain_evo_composed.predict(dataset_to_validate)

    static = static_predicted.predict
    single = single_predicted.predict
    evo_composed = evo_predicted.predict

    return single, static, evo_composed, dataset_to_validate.target
