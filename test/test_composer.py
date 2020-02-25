from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer, ComposerRequirements
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.model import XGBoost, LogRegression


def test_composer_hierarchical_chain():
    composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary_requirements=[LogRegression(), XGBoost()],
                                                 secondary_requirements=[LogRegression()])
    new_chain = composer.compose_chain(initial_chain=None,
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
    composer_requirements = ComposerRequirements(primary_requirements=[LogRegression()],
                                                 secondary_requirements=[LogRegression(), XGBoost()])
    new_chain = composer.compose_chain(data=None,initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], SecondaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[1].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[1]
    assert new_chain.nodes[0].nodes_from is None


if __name__ == '__main__':
    test_composer_hierarchical_chain()
    test_composer_flat_chain()
