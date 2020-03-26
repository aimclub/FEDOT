import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import DummyComposer, DummyChainTypeEnum, ComposerRequirements
from core.models.data import InputData
from core.models.model import LogRegression, XGBoost, train_test_data_setup
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum

random.seed(1)


@pytest.fixture()
def data_from_file():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset2.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


def make_synthetic_input(data):
    input_data = InputData(idx=np.arange(0, len(data[1])), features=data[0], target=data[1])
    return input_data


def random_target(data):
    data.target = np.array([random.choice((0, 1)) for _ in range(len(data.target))])

    return data


def compose_chain(data=None):
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=[LogRegression()],
                                                 secondary=[LogRegression(), XGBoost()])

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
    return chain


def get_roc_auc_value(chain, train_data, test_data):
    train_pred = chain.predict(new_data=train_data)
    test_pred = chain.predict(new_data=test_data)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_pred.predict)
    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_pred.predict)

    return roc_auc_value_train, roc_auc_value_test


def test_absolute_synthetic():
    data = make_classification(n_samples=1000, n_features=10, random_state=1)
    input_data = make_synthetic_input(data)

    chain = compose_chain(data=input_data)
    train_data, test_data = train_test_data_setup(input_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_test) >= 0.5
    assert abs(roc_auc_value_train) >= 0.5


def test_synthetic1_synthetic2():
    data_1 = make_classification(n_samples=1000, n_features=10, random_state=1)
    data_2 = make_classification(n_samples=1000, n_features=10, random_state=2)
    input_data_1 = make_synthetic_input(data_1)
    input_data_2 = make_synthetic_input(data_2)
    input_data_2.features = deepcopy(input_data_1.features)

    chain = compose_chain(data=input_data_1)
    train_data, _ = train_test_data_setup(input_data_1)
    _, test_data = train_test_data_setup(input_data_2)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_train) >= 0.8
    assert abs(roc_auc_value_test) >= 0.4


def test_synthetic_train_random_test():
    data_1 = make_classification(n_samples=2999, n_features=10, random_state=1)
    input_data_1 = make_synthetic_input(data_1)
    input_data_2 = random_target(input_data_1)

    chain = compose_chain(data=input_data_1)
    train_data, _ = train_test_data_setup(input_data_1)
    _, test_data = train_test_data_setup(input_data_2)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_test - 0.5) >= 0.01
    assert abs(roc_auc_value_train - 0.5) >= 0.03


def test_random_train_synthetic_test():
    data_2 = make_classification(n_samples=2999, n_features=10, random_state=1)
    input_data_2 = make_synthetic_input(data_2)
    input_data_1 = random_target(input_data_2)

    chain = compose_chain(data=input_data_1)
    train_data, _ = train_test_data_setup(input_data_1)
    _, test_data = train_test_data_setup(input_data_2)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    print(roc_auc_value_train, roc_auc_value_test)

    assert abs(roc_auc_value_test - 0.5) >= 0.01
    assert abs(roc_auc_value_train - 0.5) >= 0.03
