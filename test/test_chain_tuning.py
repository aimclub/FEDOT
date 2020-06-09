import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('data', 'scoring_train_cat.csv')
    return InputData.from_csv(os.path.join(test_file_path, file))


def get_logit_chain():
    # Chain composition
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.xgboost)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.knn)
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, final]:
        chain.add_node(node)

    return chain


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_fine_tune_primary_nodes(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    chain = get_logit_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # Chain tuning
    chain.fine_tune_primary_nodes(train_data, iterations=50)

    # After tuning prediction
    chain.fit(train_data)
    after_tuning_predicted = chain.predict(test_data)

    # Metrics
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
    aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_fine_tune_root_node(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    chain = get_logit_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # root node tuning
    chain.fine_tune_root_node(train_data, iterations=50)
    after_tun_root_node_predicted = chain.predict(test_data)

    bfr_tun_roc_auc = round(roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 3)
    aft_tun_roc_auc = round(roc_auc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 3)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc
