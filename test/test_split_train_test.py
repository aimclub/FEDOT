import random
from copy import deepcopy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.composer import DummyComposer, DummyChainTypeEnum, ComposerRequirements
from core.models.data import InputData
from core.models.model import LogRegression, XGBoost, train_test_data_setup
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

random.seed(1)
N_SAMPLES = 10000
N_FEATURES = 10
CORRECT_MODEL_AUC_THR = 0.25


def compose_chain(data: InputData) -> Chain:
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=[LogRegression()],
                                                 secondary=[LogRegression(), XGBoost()])

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
    return chain


def get_roc_auc_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(new_data=train_data)
    test_pred = chain.predict(new_data=test_data)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_pred.predict)
    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_pred.predict)

    return roc_auc_value_train, roc_auc_value_test


def get_synthetic_input_data(n_samples=10000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1])
    return input_data


def get_random_target_data(data: InputData) -> InputData:
    data_copy = deepcopy(data)
    data_copy.target = np.array([random.choice((0, 1)) for _ in range(len(data.target))])

    return data_copy


def get_auc_threshold(roc_auc_value: float) -> float:
    return abs(roc_auc_value - 0.5)


def test_correct_fit_valid():
    data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)

    chain = compose_chain(data=data)
    train_data, test_data = train_test_data_setup(data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert train_auc_thr >= CORRECT_MODEL_AUC_THR
    assert test_auc_thr >= CORRECT_MODEL_AUC_THR


def test_correct_fit_incorrect_synthetic_valid():
    train_synthetic_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    test_synthetic_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=2)
    test_synthetic_data.features = deepcopy(train_synthetic_data.features)

    chain = compose_chain(data=train_synthetic_data)
    train_data, _ = train_test_data_setup(train_synthetic_data)
    _, test_data = train_test_data_setup(test_synthetic_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_correct_fit_incorrect_random_valid():
    train_synthetic_data = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    test_random_data = get_random_target_data(train_synthetic_data)

    chain = compose_chain(data=train_synthetic_data)
    train_data, _ = train_test_data_setup(train_synthetic_data)
    _, test_data = train_test_data_setup(test_random_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_incorrect_random_fit_correct_valid():
    data_for_test = get_synthetic_input_data(N_SAMPLES, N_FEATURES, random_state=1)
    data_for_train = get_random_target_data(data_for_test)

    chain = compose_chain(data=data_for_test)
    train_data, _ = train_test_data_setup(data_for_train)
    _, test_data = train_test_data_setup(data_for_test)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr >= CORRECT_MODEL_AUC_THR
    assert train_auc_thr <= CORRECT_MODEL_AUC_THR
