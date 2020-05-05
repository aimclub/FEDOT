import datetime
import os
import random
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.optimisers.crossover import CrossoverTypesEnum
from core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from core.composer.optimisers.mutation import MutationTypesEnum
from core.composer.optimisers.regularization import RegularizationTypesEnum
from core.composer.optimisers.selection import SelectionTypesEnum
from core.composer.visualisation import ComposerVisualiser
from core.models.model import *
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root

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
                               max_lead_time: datetime.timedelta = datetime.timedelta(minutes=20),
                               gp_optimiser_params: Optional[GPChainOptimiserParameters] = None):
    dataset_to_compose = InputData.from_csv(train_file_path)
    dataset_to_validate = InputData.from_csv(test_file_path)
    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types_primary, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=[MachineLearningTasksEnum.classification,
                                                          MachineLearningTasksEnum.clustering],
                                               can_be_initial=True))

    available_model_types_secondary, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=[MachineLearningTasksEnum.classification,
                                                          MachineLearningTasksEnum.clustering],
                                               can_be_secondary=True))

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    if gp_optimiser_params:
        optimiser_parameters = gp_optimiser_params
    else:
        optimiser_parameters = GPChainOptimiserParameters(selection_types=[SelectionTypesEnum.tournament],
                                                          crossover_types=[CrossoverTypesEnum.subtree],
                                                          mutation_types=[MutationTypesEnum.simple,
                                                                          MutationTypesEnum.growth,
                                                                          MutationTypesEnum.reduce],
                                                          regularization_type=RegularizationTypesEnum.decremental)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types_primary,
        secondary=available_model_types_secondary, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time)

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, optimiser_parameters=optimiser_parameters,
                                                is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    # the choice and initialisation of the dummy_composer
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)

    chain_static = dummy_composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=True)
    chain_static.fit(input_data=dataset_to_compose, verbose=True)
    # the single-model variant of optimal chain
    single_composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.xgboost],
                                                        secondary=[])
    chain_single = DummyComposer(DummyChainTypeEnum.flat).compose_chain(data=dataset_to_compose,
                                                                        initial_chain=None,
                                                                        composer_requirements=single_composer_requirements,
                                                                        metrics=metric_function)
    chain_single.fit(input_data=dataset_to_compose, verbose=True)
    print("Composition finished")

    ComposerVisualiser.visualise(chain_static)
    ComposerVisualiser.visualise(chain_evo_composed)

    # the quality assessment for the obtained composite models
    roc_on_valid_static = calculate_validation_metric(chain_static, dataset_to_validate)
    roc_on_valid_single = calculate_validation_metric(chain_single, dataset_to_validate)
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Static ROC AUC is {round(roc_on_valid_static, 3)}')
    print(f'Single-model ROC AUC is {round(roc_on_valid_single, 3)}')

    return (roc_on_valid_evo_composed, chain_evo_composed), (chain_static, roc_on_valid_static), (
        chain_single, roc_on_valid_single)


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)
    run_credit_scoring_problem(full_path_train, full_path_test)
