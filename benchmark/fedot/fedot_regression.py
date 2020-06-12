import datetime
import random

from benchmark.benchmark_utils import get_models_hyperparameters
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.model import *
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def run_regression_problem(train_file_path, test_file_path, cur_lead_time: int = 10, vis_flag: bool = False):
    task = Task(TaskTypesEnum.regression)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    models_hyperparameters = get_models_hyperparameters()['FEDOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']

    available_model_types, _ = ModelTypesRepository(). \
        suitable_model(task_type=task.task_type)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=2, pop_size=population_size, num_of_generations=generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    single_composer_requirements = ComposerRequirements(primary=['lasso', 'ridge'],
                                                        secondary=['linear'])
    chain_static = DummyComposer(
        DummyChainTypeEnum.hierarchical).compose_chain(data=dataset_to_compose,
                                                       initial_chain=None,
                                                       composer_requirements=single_composer_requirements,
                                                       metrics=metric_function)
    chain_static.fit(input_data=dataset_to_compose, verbose=False)

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose, iterations=50)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)

    # the single-model variant of optimal chain
    single_composer_requirements = ComposerRequirements(primary=['lasso'],
                                                        secondary=[])
    chain_single = DummyComposer(DummyChainTypeEnum.flat).compose_chain(
        data=dataset_to_compose,
        initial_chain=None,
        composer_requirements=single_composer_requirements,
        metrics=metric_function)
    chain_single.fit(input_data=dataset_to_compose, verbose=False)

    static_predicted = chain_static.predict(dataset_to_validate)
    single_predicted = chain_single.predict(dataset_to_validate)
    evo_predicted = chain_evo_composed.predict(dataset_to_validate)

    static = static_predicted.predict
    single = single_predicted.predict
    evo_composed = evo_predicted.predict

    return single, static, evo_composed, dataset_to_validate.target
