import numpy as np

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from test.integration.models.test_split_train_test import get_roc_auc_value, get_synthetic_input_data


def generate_pipeline() -> Pipeline:
    node_scaling = PipelineNode('scaling')
    node_first = PipelineNode('kmeans', nodes_from=[node_scaling])
    node_second = PipelineNode('kmeans', nodes_from=[node_scaling])
    node_root = PipelineNode('logit', nodes_from=[node_first, node_second])
    pipeline = Pipeline(node_root)
    return pipeline


def test_pipeline_with_clusters_fit_correct():
    mean_roc_on_test = 0

    # mean ROC AUC is analysed because of stochastic clustering
    for _ in range(5):
        data = get_synthetic_input_data(n_samples=10000)

        pipeline = generate_pipeline()
        train_data, test_data = train_test_data_setup(data)

        pipeline.fit(input_data=train_data)
        _, roc_on_test = get_roc_auc_value(pipeline, train_data, test_data)
        mean_roc_on_test = np.mean([mean_roc_on_test, roc_on_test])

    roc_threshold = 0.5
    assert mean_roc_on_test > roc_threshold
