from core.composer.composer import DummyComposer
from core.model import XGBoost, LogRegression


def test_composer():
    composer = DummyComposer()
    new_chain = composer.compose_chain(initial_chain=None,
                                       primary_requirements=[LogRegression(), XGBoost()],
                                       secondary_requirements=[LogRegression()],
                                       metrics=None)

    assert len(new_chain.nodes) == 3
