import numpy as np
import pytest

from sklearn.datasets import load_iris
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.node import SecondaryNode
from core.models.data import InputData
from core.models.model import LogRegression
from core.composer.chain import (
    has_no_cycle,
    has_primary_nodes,
    has_no_self_cycled_nodes,
    has_no_isolated_nodes

)
from core.repository.node_types import SecondaryNodeType


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


@pytest.mark.xfail(raises=ValueError)
def test_chain_has_cycles(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3], status=SecondaryNodeType.terminal)
    y2.nodes_from.append(y4)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    has_no_cycle(chain)


def test_chain_has_no_cycles(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3], status=SecondaryNodeType.terminal)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    has_no_cycle(chain)


@pytest.mark.xfail(raises=ValueError)
def test_has_isolated_nodes(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y4])
    y6 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y5], status=SecondaryNodeType.terminal)
    y4.nodes_from.append(y6)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    has_no_isolated_nodes(chain)


@pytest.mark.xfail(raises=ValueError)
def test_multi_root_node(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3], status=SecondaryNodeType.terminal)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    assert isinstance(chain.root_node, SecondaryNode)


def test_has_primary_ndoes():
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1], status=SecondaryNodeType.terminal)
    chain.add_node(y1)
    chain.add_node(y2)
    assert has_primary_nodes(chain)


@pytest.mark.xfail(raises=ValueError)
def test_has_self_cycled_nodes():
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2], status=SecondaryNodeType.terminal)
    y2.nodes_from.append(y2)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    has_no_self_cycled_nodes(chain)


def test_validate_chain():
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y2 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y3 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y4 = NodeGenerator.primary_node(input_data=data, model=LogRegression())
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1, y2])
    y6 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3, y4])
    y7 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y5, y6], status=SecondaryNodeType.terminal)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    chain.add_node(y7)
    chain._self_validation()
