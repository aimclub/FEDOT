import logging
import os.path
from copy import deepcopy
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
                  early_stopping_generations=50,
                  cv_folds=10,

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
    print(metrics)
    # print(f'Composed ROC AUC is {round(metrics["roc_auc"], 3)}')

    if is_visualise and not predefined_model:
        print(fedot.history.get_leaderboard())
        # fedot.current_pipeline.show()

    # [0.826, 0.735, 0.754, 0.784, 0.779, 0.804, 0.79, 0.839, 0.76, 0.78]  # 0.785
    if save_prefix:
        file_name = save_prefix + '.ppl.json'
        save_path = base_path / 'openml' / file_name
        fedot.current_pipeline.save(str(save_path))

    # fedot.plot_prediction()
    return metrics


def get_preprocessed_data_folds(nfolds=(0, 1), stage='train'):
    npy_path = base_path / 'datasets'
    basename = 'credit-g'

    def load_npy(ifold: int, stage='train'):
        fname_features = f'{stage}_{basename}_fold{ifold}.npy'
        fname_target = f'{stage}y_{basename}_fold{ifold}.npy'
        features = np.load(str(npy_path / fname_features))
        targets = np.load(str(npy_path / fname_target)).astype(int)
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
                     features=features.astype(float),
                     target=_transform_targets(targets),
                     idx=np.arange(len(targets)))


def _transform_targets(targets: np.ndarray):
    targets.astype(int)
    new_targets = np.empty_like(targets).astype(str)
    new_targets[targets == 0] = 'bad'
    new_targets[targets == 1] = 'good'
    return new_targets


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
    # dummy_pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()
    output = []
    for data in data_inputs:
        data_prox = prox.obligatory_prepare_for_fit(data)
        # data_prox = prox.optional_prepare_for_fit(dummy_pipeline, data_prox)
        data_prox = prox._apply_imputation_unidata(data_prox, source_name='default')
        data_prox = prox._apply_categorical_encoding(data_prox, source_name='default')
        output.append(data_prox)
    return tuple(output)


def try_predefined(folds=None):
    amlb_path_18 = base_path / 'openml/October-09-2022,18-11-50,PM amb.ppl/amb.ppl.json'
    raw_path_17 = base_path / 'openml/October-09-2022,17-37-22,PM raw.ppl/raw.ppl.json'
    raw_path_18 = base_path / 'openml/October-09-2022,18-23-38,PM raw.ppl/raw.ppl.json'
    ppl_path = amlb_path_18
    # pipeline = Pipeline.from_serialized(source=str(ppl_path))

    pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()
    # pipeline = PipelineBuilder().add_branch('scaling', 'fast_ica').join_branches('rf').to_pipeline()

    # pipeline.show()

    train_test_raw = get_raw_data()
    metris_raw = run_classification_example(*train_test_raw, predefined_model=deepcopy(pipeline))

    for i in range(0, 10):
        train_test_pro_amb = get_preprocessed_data(nfolds=(i, i+1))
        metris_amb = run_classification_example(*train_test_pro_amb, predefined_model=deepcopy(pipeline))

        print(f'fold {i}')
        print(f'raw data: {metris_raw}')
        print(f'amb data: {metris_amb}')


def try_duplicates():
    train0, test0 = sort_data_many(*get_raw_data())
    train0p, test0p = preprocess_data(train0, test0)
    train1, test1 = sort_data_many(*get_preprocessed_data())

    features0 = train0p.features
    features1 = train1.features

    map_pairs = []
    for i, row1 in enumerate(features1):
        for j, row0 in enumerate(features0):
            if np.allclose(row0, row1):
                eq_pair = (i, j)
                map_pairs.append(eq_pair)
                print('same pair ', eq_pair)
    print(f'total dupls: {len(map_pairs)}')


def sort_data_many(*data: InputData):
    return [sort_data_by_column(d) for d in data]

def sort_data_by_column(data: InputData, *, column=4, sort_index=False) -> InputData:
    inds = data.features[:, column].argsort()

    sorted_data = deepcopy(data)
    if sort_index:
        sorted_data.idx = sorted_data.idx[inds]
    sorted_data.features = sorted_data.features[inds]
    sorted_data.target = sorted_data.target[inds]

    return sorted_data


def try_evolution(kind='raw', shuffle=False):
    train_test_raw = get_raw_data()
    train_test_pro_fdt = preprocess_data(*train_test_raw)
    train_test_pro_amb = get_preprocessed_data()

    if kind == 'raw':
        train, test = train_test_raw
    elif kind == 'amb':
        train, test = train_test_pro_amb
    elif kind == 'prx':
        train, test = train_test_pro_fdt
    else:
        raise ValueError('Unknown kind of data')

    if shuffle:
        train = train.shuffle()
        test = test.shuffle()

    print(f'Running: {kind}')
    run_classification_example(train, test, timeout=2, save_prefix=kind)


if __name__ == '__main__':
    # try_duplicates()
    # try_predefined()
    try_evolution(kind='prx')
