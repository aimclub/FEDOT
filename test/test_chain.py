import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100))
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
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[first])
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second, third])

    train_predicted = final.fit(input_data=train)

    assert train_predicted.predict.shape == train.target.shape
    assert final.cache.actual_cached_model is not None


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

    assert chain.length == 4
    assert chain.depth == 4
    assert train_predicted.predict.shape == train.target.shape