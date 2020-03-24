import os
import random

import numpy as np
import pandas as pd
import pytest

from core.composer.chain import Chain
from core.composer.composer import DummyComposer, DummyChainTypeEnum, ComposerRequirements
from core.models.data import InputData
from core.models.model import LogRegression, XGBoost
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from sklearn.metrics import roc_auc_score as roc_auc

SPLIT_RATIO = 0.8

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


@pytest.fixture()
def correct_train_random_test(data_from_file):
    data = data_from_file
    split_point = int(len(data.target) * SPLIT_RATIO)
    data.target[split_point:] = np.array([random.choice((0, 1)) for _ in range(len(data.target[split_point:]))])

    return data


@pytest.fixture()
def random_train_correct_test(data_from_file):
    data = data_from_file
    split_point = int(len(data.target) * SPLIT_RATIO)
    data.target[:split_point] = np.array([random.choice((0, 1)) for _ in range(len(data.target[:split_point]))])

    return data


@pytest.fixture()
def random_train_test(data_from_file):
    data = data_from_file
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


@pytest.mark.parametrize('data_fixture', ['correct_train_random_test'])
def test_correct_train_random_test(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    split_point = int(len(data.target) * SPLIT_RATIO)
    chain = compose_chain(data=data)
    pred = chain.predict(data)
    roc_auc_value_test = roc_auc(y_true=data.target[split_point:], y_score=pred.predict[split_point:])
    roc_auc_value_train = roc_auc(y_true=data.target[:split_point], y_score=pred.predict[:split_point])

    assert roc_auc_value_train > 0.8
    assert roc_auc_value_test > 0.45


@pytest.mark.parametrize('data_fixture', ['random_train_correct_test'])
def test_random_train_correct_test(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    split_point = int(len(data.target) * SPLIT_RATIO)
    chain = compose_chain(data=data)
    pred = chain.predict(data)
    roc_auc_value_test = roc_auc(y_true=data.target[split_point:], y_score=pred.predict[split_point:])
    roc_auc_value_train = roc_auc(y_true=data.target[:split_point], y_score=pred.predict[:split_point])

    assert roc_auc_value_train > 0.45
    assert roc_auc_value_test > 0.45


@pytest.mark.parametrize('data_fixture', ['random_train_test'])
def test_random_train_test(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    chain = compose_chain(data=data)
    pred = chain.predict(data)
    split_point = int(len(data.target) * SPLIT_RATIO)
    roc_auc_value_test = roc_auc(y_true=data.target[split_point:], y_score=pred.predict[split_point:])
    roc_auc_value_train = roc_auc(y_true=data.target[:split_point], y_score=pred.predict[:split_point])

    assert roc_auc_value_train > 0.45
    assert roc_auc_value_test > 0.45
