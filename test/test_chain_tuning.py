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


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_fine_tune_primary_nodes(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.xgboost)
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                         nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, final]:
        chain.add_node(node)

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # Chain tuning
    chain.fine_tune_primary_nodes(train_data)
    print(chain.nodes[0].model._eval_strategy)

    # After tuning prediction
    chain.fit(train_data)
    after_tuning_predicted = chain.predict(test_data)

    # Metrics
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
    aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)

    print(bfr_tun_roc_auc)
    print(aft_tun_roc_auc)

    assert aft_tun_roc_auc != bfr_tun_roc_auc
    assert list(before_tuning_predicted.predict) != list(after_tuning_predicted.predict)


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_fine_tune_root_node(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.xgboost)
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                         nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, final]:
        chain.add_node(node)

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # root node tuning preprocessing
    primary_pred = []
    for node in chain.nodes[:2]:
        pred = node.predict(train_data)
        primary_pred.append(pred.predict)

    first_feature = np.array([primary_pred[0]]).T
    second_feature = np.array([primary_pred[1]]).T

    new_features = np.concatenate((first_feature, second_feature), axis=1)
    final_data = InputData(features=new_features,
                           target=train_data.target,
                           idx=test_data.idx,
                           task_type=test_data.task_type)

    # root node tuning
    chain.fine_tune_root_node(final_data)
    after_tun_root_node_predicted = chain.predict(test_data)

    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)
    aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict)

    print(bfr_tun_roc_auc)
    print(aft_tun_roc_auc)

    assert bfr_tun_roc_auc != aft_tun_roc_auc
    assert list(before_tuning_predicted.predict) != list(after_tun_root_node_predicted.predict)
