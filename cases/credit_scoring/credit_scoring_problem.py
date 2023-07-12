import logging
import os
from pathlib import Path

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from fedot.core.utils import set_random_seed


def calculate_validation_metric(pipeline: Pipeline, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = pipeline.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               timeout: float = 5.0,
                               visualization=False,
                               target='target',
                               **composer_args):
    automl = Fedot(problem='classification',
                   timeout=timeout,
                   preset=BEST_QUALITY_PRESET_NAME,
                   logging_level=logging.DEBUG,
                   **composer_args)
    automl.fit(train_file_path, target=target)
    automl.predict(test_file_path)
    metrics = automl.get_metrics()

    if automl.history and automl.history.generations:
        print(automl.history.get_leaderboard())

    if visualization:
        automl.current_pipeline.show()

    print(f'Composed ROC AUC is {round(metrics["roc_auc_pen"], 3)}')

    return metrics["roc_auc_pen"]


def get_scoring_data():
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = fedot_project_root().joinpath(file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = fedot_project_root().joinpath(file_path_test)

    return full_path_train, full_path_test


if __name__ == '__main__':
    set_random_seed(42)

    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train,
                               full_path_test,
                               timeout=2,
                               visualization=True,
                               with_tuning=True)
