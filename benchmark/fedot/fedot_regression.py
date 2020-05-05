import datetime
import random

from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.model import *
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository
from core.repository.quality_metrics_repository import RegressionMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum


random.seed(1)
np.random.seed(1)


def run_regression_problem(train_file_path, test_file_path, cur_lead_time: int = 10, vis_flag: bool = False):
    problem_class = MachineLearningTasksEnum.regression
    dataset_to_compose = InputData.from_csv(train_file_path, task_type=problem_class)
    dataset_to_validate = InputData.from_csv(test_file_path, task_type=problem_class)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=problem_class,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=2, pop_size=10, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    single_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.lasso, ModelTypesIdsEnum.ridge],
                                                        secondary=[ModelTypesIdsEnum.linear])
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
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)

    # the single-model variant of optimal chain
    single_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.lasso],
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

    return static, single, evo_composed, dataset_to_validate.target
