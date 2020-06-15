import datetime
import random
from typing import Optional

from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.optimisers.crossover import CrossoverTypesEnum
from core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from core.composer.optimisers.mutation import MutationTypesEnum
from core.composer.optimisers.regularization import RegularizationTypesEnum
from core.composer.optimisers.selection import SelectionTypesEnum
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


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5),
                               gp_optimiser_params: Optional[GPChainOptimiserParameters] = None, pop_size=None,
                               generations=None):
    dataset_to_compose = InputData.from_csv(train_file_path)
    dataset_to_validate = InputData.from_csv(test_file_path)

    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=[MachineLearningTasksEnum.classification,
                                                          MachineLearningTasksEnum.clustering],
                                               can_be_initial=True,
                                               can_be_secondary=True))

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    if gp_optimiser_params:
        optimiser_parameters = gp_optimiser_params
    else:
        selection_types = [SelectionTypesEnum.tournament]
        crossover_types = [CrossoverTypesEnum.subtree]
        mutation_types = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
        regularization_type = RegularizationTypesEnum.decremental
        optimiser_parameters = GPChainOptimiserParameters(selection_types=selection_types,
                                                          crossover_types=crossover_types,
                                                          mutation_types=mutation_types,
                                                          regularization_type=regularization_type)
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=4,
        max_depth=3, pop_size=pop_size, num_of_generations=generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time)

    # Create GP-based composer
    composer = GPComposer()

    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, optimiser_parameters=optimiser_parameters,
                                                is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed, chain_evo_composed, composer
