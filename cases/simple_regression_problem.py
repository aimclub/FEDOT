import datetime
import os
import random

from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.data import OutputData
from core.models.model import *
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository
from core.repository.quality_metrics_repository import RegressionMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root

random.seed(1)
random.seed(1)
np.random.seed(1)

import matplotlib.pyplot as plt


def compare_plot(predicted: OutputData, dataset_to_validate: InputData):
    fig, ax = plt.subplots()
    plt.plot(dataset_to_validate.target, linewidth=1, label="Observed")
    plt.plot(predicted.predict, linewidth=1, label="Predicted")
    ax.legend()

    plt.show()


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)

    # plot results
    compare_plot(predicted, dataset_to_validate)

    # the quality assessment for the simulation results
    roc_auc_value = mse(y_true=dataset_to_validate.target,
                        y_pred=predicted.predict,
                        squared=False)
    return roc_auc_value


# the dataset was obtained from NEMO model simulation

# specify problem type
problem_class = MachineLearningTasksEnum.regression

# a dataset that will be used as a train and test set during composition
file_path_train = 'cases/data/ts/metocean_data_train.csv'
full_path_train = os.path.join(str(project_root()), file_path_train)
dataset_to_compose = InputData.from_csv(full_path_train, task_type=problem_class)

# a dataset for a final validation of the composed model
file_path_test = 'cases/data/ts/metocean_data_test.csv'
full_path_test = os.path.join(str(project_root()), file_path_test)
dataset_to_validate = InputData.from_csv(full_path_test, task_type=problem_class)

# the search of the models provided by the framework that can be used as nodes in a chain for the selected task
models_repo = ModelTypesRepository()
available_model_types, _ = models_repo.search_models(
    desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                           output_type=CategoricalDataTypesEnum.vector,
                                           task_type=problem_class,
                                           can_be_initial=True,
                                           can_be_secondary=True))

# the choice of the metric for the chain quality assessment during composition
metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

# the choice and initialisation

composer_requirements = GPComposerRequirements(
    primary=available_model_types,
    secondary=available_model_types, max_arity=2,
    max_depth=2, pop_size=10, num_of_generations=10,
    crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=3))

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
                                            metrics=metric_function, is_visualise=False)
chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)

train_prediction = chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)
print("Composition finished")

compare_plot(train_prediction, dataset_to_compose)

# the quality assessment for the obtained composite models
rmse_on_valid_static = calculate_validation_metric(chain_static, dataset_to_validate)
rmse_on_valid_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

print(f'Static RMSE is {round(rmse_on_valid_static, 3)}')
print(f'Composed RMSE is {round(rmse_on_valid_composed, 3)}')
