import logging
import os.path
from pathlib import Path

import numpy as np
from typing import Generator, Tuple

import pandas as pd

from fedot.api.main import Fedot
from fedot.core.composer.metrics import ROCAUC
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def run_classification_example(
        train_data,
        test_data,
        timeout: float = None,
        is_visualise: bool = True):

    fedot = Fedot(problem='classification',
                  metric='roc_auc',
                  timeout=timeout,
                  preset=BEST_QUALITY_PRESET_NAME,
                  with_tuning=False,
                  cv_folds=5,
                  early_stopping_generations=100,

                  seed=42,
                  n_jobs=-1,
                  logging_level=logging.INFO,
                  )

    fedot.fit(features=train_data.features, target=train_data.target)

    # fedot.fit(features=train_data_path, target='target')
    # fedot.predict(features=test_data_path)

    fedot.predict(features=test_data.features)
    metrics = fedot.get_metrics(target=test_data.target)
    print(f'Composed ROC AUC is {round(metrics["roc_auc"], 3)}')

    if is_visualise:
        print(fedot.history.get_leaderboard())
        fedot.current_pipeline.show()

    # fedot.plot_prediction()


base_path = Path('../../data/openml')


def get_preprocessed_data_folds(nfolds=10, stage='train'):
    npy_path = base_path / 'datasets'
    basename = 'credit-g'

    def load_npy(ifold: int, stage='train'):
        fname_features = f'{stage}_{basename}_fold{ifold}.npy'
        fname_target = f'{stage}y_{basename}_fold{ifold}.npy'
        features = np.load(str(npy_path / fname_features))
        targets = np.load(str(npy_path / fname_target))
        return features, targets

    all_features = []
    all_targets = []
    for ifold in range(nfolds):
        print(f'loading fold: {ifold}')
        # train_x, train_y = load_npy(ifold, stage='train')
        features, targets = load_npy(ifold, stage=stage)
        all_features.append(features)
        all_targets.append(targets)

    features = np.concatenate(all_features)
    targets = np.concatenate(all_targets)
    return InputData(task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table,
                     features=features,
                     target=targets,
                     idx=np.arange(len(targets)))


def get_preprocessed_data(nfolds=1):
    train_set = get_preprocessed_data_folds(nfolds=nfolds, stage='train')
    test_set = get_preprocessed_data_folds(nfolds=nfolds, stage='test')
    return train_set, test_set


def get_raw_data(split_ratio=0.9):
    # df = pd.read_csv(base_path / 'credit-g.csv')
    data = InputData.from_csv(base_path / 'credit-g.csv',
                              target_columns='class', index_col=None)
    train_set, test_set = train_test_data_setup(data, split_ratio=split_ratio)
    return train_set, test_set


if __name__ == '__main__':
    train_set, test_set = get_raw_data()
    # train_set, test_set = get_preprocessed_data()

    run_classification_example(train_set, test_set, timeout=1.0)
