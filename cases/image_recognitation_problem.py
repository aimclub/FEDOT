import datetime
import gzip
import os
import pickle
import random
import urllib

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.models.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_image_recognitation_problem(train_dataset: str,
                                    test_dataset: str,
                                    max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5),
                                    augmentation_flag: bool = False,
                                    is_visualise: bool = False):
    X_train, y_train = training_set
    X_test, y_test = testing_set
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_image(X_train, y_train, task=task, aug_flag=augmentation_flag)
    dataset_to_validate = InputData.from_image(X_test, y_test, task=task, aug_flag=augmentation_flag)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.log_loss)
    # metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)
    # metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.accuracy)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time)

    # composer_requirements = GPNNComposerRequirements(conv_types=conv_types, pool_types=pool_types,
    #                                                  cnn_secondary=cnn_secondary,
    #                                                  primary=nn_primary, secondary=nn_secondary, min_arity=2,
    #                                                  max_arity=2,
    #                                                  max_depth=3, pop_size=20, num_of_generations=20,
    #                                                  crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time,
    #                                                  image_size=[75, 75], train_epochs_num=2)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_optimiser_parameters(optimiser_parameters)

    # Create GP-based composer
    composer = builder.build()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                is_visualise=True)

    chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose,
                                               iterations=50)

    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    if is_visualise:
        ComposerVisualiser.visualise(chain_evo_composed)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                            dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed, chain_evo_composed


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition
    # load MNIST dataset
    mnistfile = 'mnist.pkl.gz'
    if not os.path.isfile(mnistfile):
        url = urllib.request.URLopener()
        url.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", mnistfile)

    f = gzip.open(mnistfile, 'rb')
    training_set, validation_set, testing_set = pickle.load(f, encoding='latin1')
    f.close()


    run_image_recognitation_problem(train_dataset=training_set,
                                    test_dataset=testing_set,
                                    is_visualise=True)
