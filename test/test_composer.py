from core.composer.composer import DummyComposer
from core.model import XGBoost, LogRegression


def test_composer():
    composer = DummyComposer()
    new_chain = composer.compose_chain(initial_chain=None,
                                       requirements=[LogRegression, XGBoost],
                                       metrics=None)

    assert len(new_chain.nodes) == 3
