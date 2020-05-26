import datetime
import random

from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.model import *
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum

random.seed(1)
np.random.seed(1)


def run_classification_problem(train_file_path, test_file_path, cur_lead_time: int = 10, vis_flag: bool = False):
    dataset_to_compose = InputData.from_csv(train_file_path)
    dataset_to_validate = InputData.from_csv(test_file_path)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    # the choice and initialisation of the dummy_composer
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)

    chain_static = dummy_composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_static.fit(input_data=dataset_to_compose, verbose=True)
    # the single-model variant of optimal chain
    single_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.mlp],
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
