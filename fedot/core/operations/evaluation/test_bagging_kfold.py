import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.bagging_kfold import KFoldBaggingClassifier
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root
from test.unit.pipelines.test_decompose_pipelines import get_classification_data
from sklearn.datasets import load_breast_cancer


def test_bagged_ensemble_scoring():
    # train = pd.read_csv(f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv')
    # test = pd.read_csv(f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv')

    iris = load_breast_cancer()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    train_data = InputData(
        idx=np.arange(0, len(y_train)),
        features=X_train,
        target=y_train,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    test_data = InputData(
        idx=np.arange(0, len(y_test)),
        features=X_test,
        target=y_test,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )

    pipeline = PipelineBuilder().add_node('bag_catboost').build()
    implemented_model = pipeline.fit(train_data)

    train_prediction = pipeline.predict(train_data)
    test_prediction = pipeline.predict(test_data)

    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_prediction.predict)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_prediction.predict)

    print('Train ROC-AUC score', roc_auc_value_train)
    print('Test ROC-AUC score', roc_auc_value_test)

    simple_pipeline = PipelineBuilder().add_node('catboost').build()
    implemented_model = simple_pipeline.fit(train_data)

    train_prediction = simple_pipeline.predict(train_data)
    test_prediction = simple_pipeline.predict(test_data)

    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_prediction.predict)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_prediction.predict)

    print('Simple Train ROC-AUC score', roc_auc_value_train)
    print('Simple Test ROC-AUC score', roc_auc_value_test)

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


def test_bagged_ensemble():
    train_data, test_data = get_classification_data()
    # poor params to accelerate the time
    model_names = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification, tags=['kfold_bagging']
    )

    for model_name in model_names:
        pipeline = PipelineBuilder().add_node(model_name).build()
        # TODO: Fix after solving the issue №1096
        pipeline.fit(train_data, n_jobs=-1)
        predicted_output = pipeline.predict(test_data, output_mode='labels')
        metric = roc_auc(test_data.target, predicted_output.predict)

        assert isinstance(pipeline, Pipeline)
        assert predicted_output.predict.shape[0] == 240
        assert metric > 0.5