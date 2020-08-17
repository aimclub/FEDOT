import datetime
import os
import random

from core.composer.metrics import AccuracyScore
from cases.credit_scoring_problem import calculate_validation_metric
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData, split_train_test
from core.repository.model_types_repository import (
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.tasks import Task, TaskTypesEnum
from core.utils import project_root

import numpy as np

random.seed(1)
np.random.seed(1)


def run_grasp_robustness_problem(dataset_path, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=60),
                                 is_visualise=False):
    task = Task(TaskTypesEnum.classification)
    dataset = InputData.from_csv(dataset_path, headers=['measurement_number'], task=task, target_header='robustness')

    # this is a sensible grasp threshold for stability
    good_grasp_threshold = 100

    # discretizing the grasp quality for stable or unstable grasps
    dataset.target = np.array([int(i > good_grasp_threshold) for i in dataset.target])

    # decreasing number of dataset lines
    dataset, _ = split_train_test(dataset, split_ratio=0.05)

    # split dataset to train and test sets
    dataset_to_compose, dataset_to_validate = split_train_test(dataset)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=10, num_of_generations=20,
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
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    acc_on_valid_evo_composed = AccuracyScore.get_value(chain_evo_composed, dataset_to_validate)
    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')
    print(f'Composed accuracy is {acc_on_valid_evo_composed}')

    return [chain_evo_composed, roc_on_valid_evo_composed, acc_on_valid_evo_composed]


if __name__ == '__main__':
    # the dataset was obtained from https://www.kaggle.com/ugocupcic/grasping-dataset

    # dataset path definition
    dataset_path = 'cases/data/robotics/dataset.csv'
    full_dataset_path = os.path.join(str(project_root()), dataset_path)

    # run composition
    res = run_grasp_robustness_problem(full_dataset_path, is_visualise=True)
