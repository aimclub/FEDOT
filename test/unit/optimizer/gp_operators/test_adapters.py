import numpy as np

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.graph import OptNode

from fedot.core.pipelines.adapters import PipelineAdapter
from test.unit.dag.test_graph_utils import find_first
from test.unit.optimizer.gp_operators.test_gp_operators import pipeline_with_custom_parameters
from test.unit.tasks.test_regression import get_synthetic_regression_data


def generate_so_complex_pipeline():
    node_imp = PrimaryNode('simple_imputation')
    node_lagged = SecondaryNode('lagged', nodes_from=[node_imp])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_ridge])
    node_pca = SecondaryNode('pca', nodes_from=[node_decompose])
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge, node_pca])
    pipeline = Pipeline(node_final)
    return pipeline


def test_pipeline_adapters_params_correct():
    """ Checking the correct conversion of hyperparameters in nodes when nodes
    are passing through adapter
    """
    init_alpha = 12.1
    pipeline = pipeline_with_custom_parameters(init_alpha)

    # Convert into OptGraph object
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    # Get Pipeline object back
    restored_pipeline = adapter.restore(opt_graph)
    # Get hyperparameter value after pipeline restoration
    restored_alpha = restored_pipeline.root_node.parameters['alpha']
    assert np.isclose(init_alpha, restored_alpha)


def test_preds_before_and_after_convert_equal():
    """ Check if the pipeline predictions change before and after conversion
    through the adapter
    """
    init_alpha = 12.1
    pipeline = pipeline_with_custom_parameters(init_alpha)

    # Generate data
    input_data = get_synthetic_regression_data(n_samples=10, n_features=2,
                                               random_state=2021)
    # Init fit
    pipeline.fit(input_data)
    init_preds = pipeline.predict(input_data)

    # Convert into OptGraph object
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    restored_pipeline = adapter.restore(opt_graph)

    # Restored pipeline fit
    restored_pipeline.fit(input_data)
    restored_preds = restored_pipeline.predict(input_data)

    assert np.array_equal(init_preds.predict, restored_preds.predict)


def test_no_opt_or_graph_nodes_after_adapt_so_complex_graph():
    adapter = PipelineAdapter()
    pipeline = generate_so_complex_pipeline()
    adapter.adapt(pipeline)

    assert not find_first(pipeline, lambda n: type(n) in (GraphNode, OptNode))
