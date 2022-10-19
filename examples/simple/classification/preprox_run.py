import logging
import random
from copy import deepcopy
from typing import Optional

import numpy as np

from examples.simple.classification.preprox import base_path, get_raw_data, preprocess_data, get_preprocessed_data
from fedot.api.main import Fedot
from fedot.core.constants import BEST_QUALITY_PRESET_NAME
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


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


def try_predefined(folds=None):
    amlb_path_18 = base_path / 'openml/October-09-2022,18-11-50,PM amb.ppl/amb.ppl.json'
    raw_path_17 = base_path / 'openml/October-09-2022,17-37-22,PM raw.ppl/raw.ppl.json'
    raw_path_18 = base_path / 'openml/October-09-2022,18-23-38,PM raw.ppl/raw.ppl.json'
    ppl_path = amlb_path_18
    # pipeline = Pipeline.from_serialized(source=str(ppl_path))

    # pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()
    pipeline = PipelineBuilder().add_branch('scaling', 'fast_ica').join_branches('rf').to_pipeline()

    # pipeline.show()

    train_test_raw = get_raw_data()
    metris_raw = run_classification_example(*train_test_raw, predefined_model=deepcopy(pipeline))
    train_test_pro_fdt = preprocess_data(*train_test_raw)
    metris_prx= run_classification_example(*train_test_pro_fdt, predefined_model=deepcopy(pipeline))
    print(f'raw data: {metris_raw}')
    print(f'prx data: {metris_prx}')

    metris_amb_all_rocauc = []
    # for i in range(0, 10):
    for i in [0] * 10:
        train_test_pro_amb = get_preprocessed_data(nfolds=(i, i+1), npy_path=base_path / 'datasets')
        # train_test_pro_amb = get_preprocessed_data(nfolds=(i, i+1), npy_path=base_path_npy)
        metris_amb = run_classification_example(*train_test_pro_amb, predefined_model=deepcopy(pipeline))
        metris_amb_all_rocauc.append(metris_amb["roc_auc"])
        print(f'amb data (fold {i}): {metris_amb}')
    avg_metri = np.mean(metris_amb_all_rocauc)
    print(f'amb data, mean metric on 10 folds: {avg_metri}')


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
        train.shuffle()
        test.shuffle()

    print(f'Running: {kind}')
    return run_classification_example(train, test, timeout=30, save_prefix=kind)


if __name__ == '__main__':
    random.seed(444)
    np.random.seed(444)

    # try_duplicates()
    try_predefined()

    # mamb = try_evolution(kind='amb', shuffle=True)
    # mraw = try_evolution(kind='raw', shuffle=True)
    # print('amb:', mamb)
    # print('raw:', mraw)
