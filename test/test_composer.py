from core.composer.composer import DummyComposer
from core.model import XGBoost, LogRegression
from core.node import PrimaryNode, SecondaryNode


def test_composer():
    composer = DummyComposer()
    new_chain = composer.compose_chain(initial_chain=None,
                                       primary_requirements=[LogRegression(), XGBoost()],
                                       secondary_requirements=[LogRegression()],
                                       metrics=None)

    assert len(new_chain.nodes) == 3
    assert isinstance(new_chain.nodes[0], PrimaryNode)
    assert isinstance(new_chain.nodes[1], PrimaryNode)
    assert isinstance(new_chain.nodes[2], SecondaryNode)
    assert new_chain.nodes[2].nodes_from[0] is new_chain.nodes[0]
    assert new_chain.nodes[2].nodes_from[1] is new_chain.nodes[1]
    assert new_chain.nodes[1].nodes_from is None
