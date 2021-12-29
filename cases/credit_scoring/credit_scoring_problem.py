import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(pipeline: Pipeline, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = pipeline.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               timeout: float = 5.0,
                               is_visualise=False,
                               with_tuning=False,
                               target='target'):

    automl = Fedot(problem='classification', timeout=timeout, verbose_level=4,
                   preset='best_quality', composer_params={'with_tuning': with_tuning})
    automl.fit(train_file_path, target=target)
    predict = automl.predict(test_file_path)
    metrics = automl.get_metrics()

    if is_visualise:
        automl.current_pipeline.show()

    print(f'Composed ROC AUC is {round(metrics["roc_auc"], 3)}')

    return metrics["roc_auc"]


def get_scoring_data():
    # the dataset was obtained from https://www.kaggle.com/c/GiveMeSomeCredit

    # a dataset that will be used as a train and test set during composition

    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    return full_path_train, full_path_test


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train,
                               full_path_test,
                               timeout=5,
                               is_visualise=True,
                               with_tuning=True)
