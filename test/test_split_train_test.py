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


def convert_to_input_data(data):
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


def get_synthetic_data(random_state):
    synthetic_data = make_classification(n_samples=1000, n_features=10, random_state=random_state)
    return synthetic_data


def test_one_synthetic_dataset():
    data = get_synthetic_data(1)
    input_data = convert_to_input_data(data)

    chain = compose_chain(data=input_data)
    train_data, test_data = train_test_data_setup(input_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)
    print(roc_auc_value_train)
    print(roc_auc_value_test)

    assert abs(roc_auc_value_test - 0.5) >= 0.25
    assert abs(roc_auc_value_train - 0.5) >= 0.25


def test_two_synthetic_dataset():
    first_synthetic_data = get_synthetic_data(1)
    second_synthetic_data = get_synthetic_data(2)
    first_input_data = convert_to_input_data(first_synthetic_data)
    second_input_data = convert_to_input_data(second_synthetic_data)
    second_input_data.features = deepcopy(first_input_data.features)

    chain = compose_chain(data=first_input_data)
    train_data, _ = train_test_data_setup(first_input_data)
    _, test_data = train_test_data_setup(second_input_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_test - 0.5) >= 0.01
    assert abs(roc_auc_value_train - 0.5) >= 0.25


def test_synthetic_train_random_test():
    first_synthetic_data = get_synthetic_data(1)
    first_input_data = convert_to_input_data(first_synthetic_data)
    second_input_data = random_target(first_input_data)

    chain = compose_chain(data=first_input_data)
    train_data, _ = train_test_data_setup(first_input_data)
    _, test_data = train_test_data_setup(second_input_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_test - 0.5) >= 0.01
    assert abs(roc_auc_value_train - 0.5) >= 0.03


def test_random_train_synthetic_test():
    first_synthetic_data = get_synthetic_data(1)
    first_input_data = convert_to_input_data(first_synthetic_data)
    second_input_data = random_target(first_input_data)

    chain = compose_chain(data=first_input_data)
    train_data, _ = train_test_data_setup(second_input_data)
    _, test_data = train_test_data_setup(first_input_data)

    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(chain, train_data, test_data)

    assert abs(roc_auc_value_test - 0.5) >= 0.01
    assert abs(roc_auc_value_train - 0.5) >= 0.03
