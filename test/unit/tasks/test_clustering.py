import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import train_test_data_setup
from test.unit.models.test_split_train_test import get_roc_auc_value, get_synthetic_input_data


def generate_chain() -> Chain:
    node_scaling = PrimaryNode('scaling')
    node_first = SecondaryNode('kmeans', nodes_from=[node_scaling])
    node_second = SecondaryNode('kmeans', nodes_from=[node_scaling])
    node_root = SecondaryNode('logit', nodes_from=[node_first, node_second])
    chain = Chain(node_root)
    return chain


def test_chain_with_clusters_fit_correct():
    mean_roc_on_test = 0

    # mean ROC AUC is analysed because of stochastic clustering
    for _ in range(5):
        data = get_synthetic_input_data(n_samples=10000)

        chain = generate_chain()
        train_data, test_data = train_test_data_setup(data)

        chain.fit(input_data=train_data)
        _, roc_on_test = get_roc_auc_value(chain, train_data, test_data)
        mean_roc_on_test = np.mean([mean_roc_on_test, roc_on_test])

    roc_threshold = 0.5
    assert mean_roc_on_test > roc_threshold
