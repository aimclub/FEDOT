import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root

random.seed(1)
np.random.seed(1)

def run_credit_scoring_problem():
    train_path = Path(fedot_project_root(), 'cases', 'train_australian_fold0.npy')
    train_pathy = Path(fedot_project_root(), 'cases', 'trainy_australian_fold0.npy')
    train_arr = np.load(train_path)

    train_data = pd.DataFrame(train_arr)
    train_data['target'] = pd.DataFrame(np.load(train_pathy).tolist())

    automl = Fedot(problem='classification', timeout=5, logging_level=logging.INFO,
                   preset=BEST_QUALITY_PRESET_NAME)
    automl.fit(train_data)


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
    run_credit_scoring_problem()
