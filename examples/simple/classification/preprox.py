import logging
import os.path
from pathlib import Path

import numpy as np
from typing import Generator, Tuple, Optional

import pandas as pd

from fedot.api.main import Fedot
from fedot.core.composer.metrics import ROCAUC
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root
from fedot.preprocessing.preprocessing import DataPreprocessor


base_path = Path('../../data/openml')


def run_classification_example(
        train_data,
        test_data,
        timeout: float = None,
        is_visualise: bool = True,
        save_prefix: Optional[str] = None,
        predefined_model: Optional[Pipeline] = None):

    fedot = Fedot(problem='classification',
                  metric='roc_auc',
                  timeout=timeout,
                  preset=BEST_QUALITY_PRESET_NAME,
                  with_tuning=False,
                  early_stopping_generations=100,

                  seed=42,
                  n_jobs=8,
                  logging_level=logging.INFO,
                  )

    fedot.fit(features=train_data.features, target=train_data.target,
              predefined_model=predefined_model)

    # fedot.fit(features=train_data_path, target='target')
    # fedot.predict(features=test_data_path)

    fedot.predict(features=test_data.features)
    metrics = fedot.get_metrics(target=test_data.target)
    print(f'Composed ROC AUC is {round(metrics["roc_auc"], 3)}')

    if is_visualise and not predefined_model:
        print(fedot.history.get_leaderboard())
        fedot.current_pipeline.show()

    if save_prefix:
        file_name = save_prefix + '.ppl.json'
        save_path = base_path / 'openml' / file_name
        fedot.current_pipeline.save(str(save_path))

    # fedot.plot_prediction()
    return metrics


def get_preprocessed_data_folds(nfolds=(0, 10), stage='train'):
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
    for ifold in range(*nfolds):
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


def get_preprocessed_data(nfolds=(0, 1)):
    train_set = get_preprocessed_data_folds(nfolds=nfolds, stage='train')
    test_set = get_preprocessed_data_folds(nfolds=nfolds, stage='test')
    return train_set, test_set


def get_raw_data(split_ratio=0.9):
    # df = pd.read_csv(base_path / 'credit-g.csv')
    data = InputData.from_csv(base_path / 'credit-g.csv',
                              target_columns='class', index_col=None)
    train_set, test_set = train_test_data_setup(data, split_ratio=split_ratio)
    return train_set, test_set


def preprocess_data(*data_inputs):
    prox = DataPreprocessor()
    dummy_pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()
    output = []
    for data in data_inputs:
        data_prox = prox.obligatory_prepare_for_fit(data)
        data_prox = prox.optional_prepare_for_fit(dummy_pipeline, data_prox)
        output.append(data_prox)
    return tuple(output)


def try_predefined():
    pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()

    train_test_raw = get_raw_data()
    train_test_pro_amb = get_preprocessed_data()

    metris_raw = run_classification_example(*train_test_raw, predefined_model=pipeline)
    metris_amb = run_classification_example(*train_test_pro_amb, predefined_model=pipeline)

    print(f'raw data: {metris_raw}\n'
          f'amb data: {metris_amb}')


def try_evolution(is_raw=True):
    train_test_raw = get_raw_data()

    train_test_pro_fdt = preprocess_data(*train_test_raw)
    train_test_pro_amb = get_preprocessed_data(nfolds=(6, 7))

    if is_raw:
        prefix = 'raw'
        train, test = train_test_raw
    else:
        prefix = 'amb'
        train, test = train_test_pro_amb

    prefix = 'prx'
    train, test = train_test_pro_fdt

    print('Running: ', prefix)
    run_classification_example(train, test, timeout=1, save_prefix=prefix)


if __name__ == '__main__':
    # try_predefined()
    try_evolution(is_raw=True)
