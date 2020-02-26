import numpy as np

from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.node import PrimaryNode, SecondaryNode, InputData
from core.models.model import XGBoost, LogRegression
from core.composer.composer import ComposerRequirements


def test_composer_hierarchical_chain():
    composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    empty_data = InputData(np.zeros(1), np.zeros(1), np.zeros(1))
    composer_requirements = ComposerRequirements(primary_requirements=[LogRegression(), XGBoost()],
                                       secondary_requirements=[LogRegression()])
    new_chain = composer.compose_chain(data=empty_data,
                                       initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], PrimaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[1] is new_chain.nodes[1]
    assert new_chain.nodes[1].nodes_from is None


def test_composer_flat_chain():
    composer = DummyComposer(DummyChainTypeEnum.flat)
    empty_data = InputData(np.zeros(1), np.zeros(1), np.zeros(1))
    composer_requirements = ComposerRequirements(secondary_requirements=[LogRegression(), XGBoost()],
                                       primary_requirements=[LogRegression()])
    new_chain = composer.compose_chain(data=empty_data,
                                       initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], SecondaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[1].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[1]
    assert new_chain.nodes[0].nodes_from is None
