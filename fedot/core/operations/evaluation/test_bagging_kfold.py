import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.bagging_kfold import KFoldBaggingClassifier
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def test_bagged_ensemble_scoring():
    train = pd.read_csv(f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv')
    test = pd.read_csv(f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv')

    train_data = InputData(
        idx=np.arange(0, len(train.target)),
        features=np.array(train.drop(['target'], axis=1)),
        target=np.array(train.target),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    test_data = InputData(
        idx=np.arange(0, len(test.target)),
        features=np.array(test.drop(['target'], axis=1)),
        target=np.array(test.target),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    # base_estimator = CatBoostClassifier(
    #     allow_writing_files=False,
    #     verbose=False
    # )

    base_estimator = CatBoostClassifier()

    bclf = KFoldBaggingClassifier(
        base_estimator=base_estimator,
        n_layers=2,
        n_repeats=5,
        k_fold=3
    )

    implemented_model = bclf.fit(X=train_data.features, y=train_data.target)

    train_prediction = bclf.predict(X=train_data.features)
    test_prediction = bclf.predict(X=test_data.features)

    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_prediction)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_prediction)

    print('Train ROC-AUC score', roc_auc_value_train)
    print('Test ROC-AUC score', roc_auc_value_test)

    assert implemented_model is not None
    assert test_prediction is not None
    assert test_prediction.shape[0] == test_data.target.shape[0]
    assert roc_auc_value_test >= 0.5


def test_prob_sum():
    repeats = 10
    folds = 15
    models = 15
    numbers_of_target = 1334

    class_probs = np.random.random(size=(repeats, folds, models, numbers_of_target, 1))
    test_rand_oof_probs = np.concatenate((class_probs, 1 - class_probs), axis=-1)

    repeat_0_probs_fold_0 = [
        [[0.3, 0.7], [0.25, 0.75], [0.2, 0.8]], # Y^{j=0}_{m=0, i=0}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]  # Y^{j=0}_{m=1, i=0}
    ]

    repeat_0_probs_fold_1 = [
        [[0.67, 0.33], [0.72, 0.28], [0.79, 0.21]], # Y^{j=1}_{m=0, i=0}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]      # Y^{j=1}_{m=1, i=0}
    ]

    repeat_1_probs_fold_0 = [
        [[0.1, 0.9], [0.25, 0.75], [0.2, 0.8]], # Y^{j=0}_{m=0, i=1}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]  # Y^{j=0}_{m=1, i=1}
    ]

    repeat_1_probs_fold_1 = [
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]], # Y^{j=1}_{m=0, i=1}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]  # Y^{j=1}_{m=1, i=1}
    ]

    repeat_2_probs_fold_0 = [
        [[0.15, 0.85], [0.25, 0.75], [0.2, 0.8]], # Y^{j=0}_{m=0, i=2}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]    # Y^{j=0}_{m=1, i=2}
    ]

    repeat_2_probs_fold_1 = [
        [[0.9, 0.1], [0.75, 0.25], [0.8, 0.2]], # Y^{j=1}_{m=0, i=2}
        [[0.7, 0.3], [0.75, 0.25], [0.8, 0.2]]  # Y^{j=1}_{m=1, i=2}
    ]

    oof_probs = [
        [repeat_0_probs_fold_0, repeat_0_probs_fold_1],
        [repeat_1_probs_fold_0, repeat_1_probs_fold_1],
        [repeat_2_probs_fold_0, repeat_2_probs_fold_1],
    ]

    # (n_repeats, model, n_fold, n_features, probs)
    oof_probs = np.array(oof_probs)

    Y_j_0_m_0 = np.array([
        oof_probs[0, 0, 0, :, :],   # Y^{j=0}_{m=0, i=0} p1=0.3
        oof_probs[1, 0, 0, :, :],   # Y^{j=0}_{m=0, i=1} p1=0.1
        oof_probs[2, 0, 0, :, :]    # Y^{j=0}_{m=0, i=2} p1=0.15
    ])

    sum_j_0_m_0 = np.sum(Y_j_0_m_0, axis=0)
    avg_j_0_m_0 = sum_j_0_m_0 / repeats
    preds_j_0_m_0 = np.argmax(avg_j_0_m_0, axis=1)

    Y_j_1_m_0 = np.array([
        oof_probs[0, 1, 0, :, :],   # Y^{j=1}_{m=0, i=0} p1=0.3
        oof_probs[1, 1, 0, :, :],   # Y^{j=1}_{m=0, i=1} p1=0.3
        oof_probs[2, 1, 0, :, :]    # Y^{j=1}_{m=0, i=2} p1=0.1
    ])

    sum_j_1_m_0 = np.sum(Y_j_1_m_0, axis=0)
    avg_j_1_m_0 = sum_j_1_m_0 / repeats
    preds_j_1_m_0 = np.argmax(avg_j_1_m_0, axis=1)

    preds_m_0 = np.array([preds_j_0_m_0, preds_j_1_m_0]).reshape(-1)

    assert oof_probs.shape == test_rand_oof_probs.shape == (3, 2, 2, 3, 2)
    assert sum_j_0_m_0.shape == np.sum(test_rand_oof_probs[:, 0, 0, :, :], axis=0).shape
    assert avg_j_0_m_0.shape == (np.sum(test_rand_oof_probs[:, 0, 0, :, :], axis=0) / repeats).shape
    #
    preds_per_chunk = np.argmax(np.sum(test_rand_oof_probs[:, :, 0, :, :], axis=0) / repeats, axis=2)
    y_hat_m = np.array([np.argmax(np.unique(preds, return_counts=True)[1]) for preds in preds_per_chunk.T])
    unique_class, counts = np.unique(preds_per_chunk.T, return_counts=True, axis=-1)
    y_hat_m = np.argmax(counts, axis=-1)

    assert preds_m_0.shape == (np.argmax(np.sum(test_rand_oof_probs[:, :, 0, :, :], axis=0) / repeats, axis=2).reshape(-1)).shape
