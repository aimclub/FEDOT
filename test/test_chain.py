import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.model import LogRegression


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
def test_models_sequence(data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    results = y4.apply()
    assert y4.cached_result.cached_output.size == data.target.size
    assert len(y4.cached_result.last_parents_ids) == 2
    assert len(results.predict) == len(data.target)


def test_models_chain_nested(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.predict(data)
    assert chain.length == 4
    assert chain.depth == 3
    assert len(results.predict) == len(data.target)


def test_models_chain_seq(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])

    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.predict(data)
    assert chain.length == 4
    assert chain.depth == 4
    assert len(results.predict) == len(data.target)

