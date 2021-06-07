from fedot.explainability.chain_explainer import ChainExplainer
from test.unit.chains.test_node_cache import chain_third
from test.unit.chains.test_node_cache import data_setup

setup = data_setup


def test_chain_explainer(data_setup):
    chain = chain_third()
    train, _ = data_setup
    chain.fit(train)
    explainer = ChainExplainer(chain, train)
    shap_explanation = explainer.explain_with_shap()
    assert shap_explanation is not None


