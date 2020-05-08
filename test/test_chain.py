import os
from copy import deepcopy
from random import seed

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum

seed(1)
np.random.seed(1)


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task_type=MachineLearningTasksEnum.classification)
    return data


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_nodes_sequence_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train, _ = train_test_data_setup(data)
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.lda,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.qda,
                                         nodes_from=[first])
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.knn,
                                         nodes_from=[second, third])

    train_predicted = final.fit(input_data=train)

    assert final.descriptive_id == (
        '((/n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.lda_defaultparams;;(/'
        'n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.qda_defaultparams;)/'
        'n_ModelTypesIdsEnum.knn_defaultparams')

    assert train_predicted.predict.shape == train.target.shape
    assert final.cache.actual_cached_state is not None


def test_chain_hierarchy_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[first])
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    train_predicted = chain.fit(input_data=train, use_cache=False)

    assert chain.root_node.descriptive_id == (
        '((/n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams;;(/'
        'n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams')

    assert chain.length == 4
    assert chain.depth == 3
    assert train_predicted.predict.shape == train.target.shape


def test_chain_sequential_fit_correct(data_setup):
    data = data_setup
    train, _ = train_test_data_setup(data)

    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second])
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    train_predicted = chain.fit(input_data=train, use_cache=False)

    assert chain.root_node.descriptive_id == (
        '(((/n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams;)/'
        'n_ModelTypesIdsEnum.logit_defaultparams')

    assert chain.length == 4
    assert chain.depth == 4
    assert train_predicted.predict.shape == train.target.shape


def test_secondary_nodes_is_invariant_to_inputs_order(data_setup):
    data = data_setup
    train, test = train_test_data_setup(data)
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.lda)
    third = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.knn)
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                         nodes_from=[first, second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    first = deepcopy(first)
    second = deepcopy(second)
    third = deepcopy(third)
    final_shuffled = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                                  nodes_from=[third, first, second])

    chain_shuffled = Chain()
    # change order of nodes in list
    for node in [final_shuffled, third, first, second]:
        chain_shuffled.add_node(node)

    train_predicted = chain.fit(input_data=train)

    train_predicted_shuffled = chain_shuffled.fit(input_data=train)

    # train results should be invariant
    assert chain.root_node.descriptive_id == chain_shuffled.root_node.descriptive_id
    assert all(np.equal(train_predicted.predict, train_predicted_shuffled.predict))

    test_predicted = chain.predict(input_data=test)
    test_predicted_shuffled = chain_shuffled.predict(input_data=test)

    # predict results should be invariant
    assert all(np.equal(test_predicted.predict, test_predicted_shuffled.predict))

    # change parents order for the nodes fitted chain
    nodes_for_change = chain.nodes[3].nodes_from
    chain.nodes[3].nodes_from = [nodes_for_change[2], nodes_for_change[0], nodes_for_change[1]]
    chain.nodes[3].cache.clear()
    chain.fit(train)
    test_predicted_re_shuffled = chain.predict(input_data=test)

    # predict results should be invariant
    assert all(np.equal(test_predicted.predict, test_predicted_re_shuffled.predict))
