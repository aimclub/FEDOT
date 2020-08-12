import datetime
import os
import random
from sklearn.model_selection import train_test_split

from core.composer.metrics import AccuracyScore
from cases.credit_scoring_problem import calculate_validation_metric
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum
from core.utils import project_root

import numpy as np

random.seed(1)
np.random.seed(1)


def csv_preparation(dataset_path):
    dataset = np.loadtxt(dataset_path, skiprows=1, usecols=range(1, 30), delimiter=",")
    with open(dataset_path, 'r') as f:
        header = f.readline()
    header = header.strip("\n").split(',')
    header = [i.strip(" ") for i in header]
    saved_cols = []
    for index, col in enumerate(header[1:]):
        if ("vel" in col) or ("eff" in col):
            saved_cols.append(index)
    new_X = []
    for x in dataset:
        new_X.append([x[i] for i in saved_cols])
    X = np.array(new_X)
    Y = np.array(dataset[:, 0]).reshape((X.shape[0], 1))

    # decreasing the number of lines to check time execution
    arr = np.hstack((X, Y))
    np.random.shuffle(arr)
    arr = arr[0:50000, :]

    seed = 7
    np.random.seed(seed)
    X_train, X_test, Y_train, Y_test = train_test_split(arr[:, 0:-1], arr[:, -1], test_size=0.20, random_state=seed)

    # this is a sensible grasp threshold for stability
    GOOD_GRASP_THRESHOLD = 100

    # we're also storing the best and worst grasps of the test set to do some sanity checks on them
    itemindex = np.where(Y_test > 1.05 * GOOD_GRASP_THRESHOLD)
    best_grasps = X_test[itemindex[0]]
    itemindex = np.where(Y_test <= 0.95 * GOOD_GRASP_THRESHOLD)
    bad_grasps = X_test[itemindex[0]]

    # discretizing the grasp quality for stable or unstable grasps
    Y_train = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in Y_train]).reshape((Y_train.shape[0], 1))
    Y_test = np.array([int(i > GOOD_GRASP_THRESHOLD) for i in Y_test]).reshape((Y_test.shape[0], 1))
    data_train = np.hstack((np.arange(1, X_train.shape[0] + 1).reshape((X_train.shape[0], 1)), X_train, Y_train))
    data_test = np.hstack((np.arange(1, X_test.shape[0] + 1).reshape((X_test.shape[0], 1)), X_test, Y_test))

    # dump appropriate arrays to .\cases\data\robotics
    file_path_train = 'cases/data/robotics/robotics_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    file_path_test = 'cases/data/robotics/robotics_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)
    np.savetxt(full_path_train, data_train, delimiter=",")
    np.savetxt(full_path_test, data_test, delimiter=",")


def run_grasp_robustness_problem(train_file_path, test_file_path,
                                 max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5),
                                 is_visualise=False):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=10, num_of_generations=15,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time)

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function,
                                                is_visualise=False)

    chain_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose,
                                               iterations=50)

    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    if is_visualise:
        ComposerVisualiser.visualise(chain_evo_composed)

    # the quality assessment for the obtained composite models
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                            dataset_to_validate)

    acc_on_valid_evo_composed = AccuracyScore.get_value(chain_evo_composed,
                                                            dataset_to_validate)
    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed ROC AUC is {acc_on_valid_evo_composed}')

    return roc_on_valid_evo_composed


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/ugocupcic/grasping-dataset

    # download and preprocessing dataset
    dataset_path = 'cases/data/robotics/dataset.csv'
    full_dataset_path = os.path.join(str(project_root()), dataset_path)
    csv_preparation(full_dataset_path)

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/robotics/robotics_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/robotics/robotics_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_grasp_robustness_problem(full_path_train, full_path_test, is_visualise=True)
