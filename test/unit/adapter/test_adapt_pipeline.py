from copy import deepcopy
from random import choice

import numpy as np
import pytest

from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptNode

from fedot.core.operations.operation import Operation
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from test.unit.dag.test_graph_utils import find_first
from test.unit.tasks.test_regression import get_synthetic_regression_data


def get_pipelines():
    one_node_pipeline = PipelineBuilder() \
        .add_sequence('logit') \
        .build()
    linear_pipeline = PipelineBuilder() \
        .add_sequence('logit', 'logit', 'logit') \
        .build()
    branching_structure = PipelineBuilder() \
        .add_node('logit') \
        .add_branch('logit', 'logit') \
        .join_branches('logit') \
        .build()
    branching_structure2 = PipelineBuilder() \
        .add_node('logit') \
        .add_branch('logit', 'logit') \
        .grow_branches('logit', None) \
        .join_branches('logit') \
        .add_node('logit') \
        .build()
    node_a = PipelineNode('logit')
    node_b = PipelineNode('logit', nodes_from=[node_a])
    node_c = PipelineNode('logit', nodes_from=[node_b, node_a])
    skip_connection_structure = Pipeline(node_c)

    return [one_node_pipeline, linear_pipeline,
            branching_structure, branching_structure2,
            skip_connection_structure]


def generate_so_complex_pipeline():
    node_imp = PipelineNode('simple_imputation')
    node_lagged = PipelineNode('lagged', nodes_from=[node_imp])
    node_ridge = PipelineNode('ridge', nodes_from=[node_lagged])
    node_decompose = PipelineNode('decompose', nodes_from=[node_lagged, node_ridge])
    node_pca = PipelineNode('pca', nodes_from=[node_decompose])
    node_final = PipelineNode('ridge', nodes_from=[node_ridge, node_pca])
    pipeline = Pipeline(node_final)
    return pipeline


def pipeline_with_custom_parameters(alpha_value):
    node_scaling = PipelineNode('scaling')
    node_norm = PipelineNode('normalization')
    node_dtreg = PipelineNode('dtreg', nodes_from=[node_scaling])
    node_lasso = PipelineNode('lasso', nodes_from=[node_norm])
    node_final = PipelineNode('ridge', nodes_from=[node_dtreg, node_lasso])
    node_final.parameters = {'alpha': alpha_value}
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


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_pipelines_adapt_properly(pipeline):
    adapter = PipelineAdapter()
    verifier = GraphVerifier(DEFAULT_DAG_RULES)

    assert all(isinstance(node, PipelineNode) for node in pipeline.nodes)
    assert _check_nodes_references_correct(pipeline)
    assert verifier(pipeline)

    opt_graph = adapter.adapt(pipeline)

    assert all(isinstance(node, OptNode) for node in opt_graph.nodes)  # checking strict type equality!
    assert _check_nodes_references_correct(opt_graph)
    assert verifier(opt_graph)


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_adapted_has_same_structure(pipeline):
    adapter = PipelineAdapter()

    opt_graph = adapter.adapt(pipeline)

    # assert graph structures are same
    assert pipeline.descriptive_id == opt_graph.descriptive_id


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_adapted_and_restored_are_equal(pipeline):
    adapter = PipelineAdapter()

    opt_graph = adapter.adapt(pipeline)
    restored_pipeline = adapter.restore(opt_graph)

    # assert 2-way mapping doesn't change the structure
    assert pipeline.descriptive_id == restored_pipeline.descriptive_id
    # assert that new pipeline is a different object
    assert id(pipeline) != id(restored_pipeline)


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_changes_to_transformed_dont_affect_origin(pipeline):
    adapter = PipelineAdapter()

    original_pipeline = deepcopy(pipeline)
    opt_graph = adapter.adapt(pipeline)

    # before change they're equal
    assert pipeline.descriptive_id == opt_graph.descriptive_id

    changed_node = choice(opt_graph.nodes)
    changed_node.content['name'] = 'linear'
    changed_node.content['params'].update({'new_hyperparam': 39})

    # assert that changes to the adapted pipeline don't affect original graph
    assert pipeline.descriptive_id != opt_graph.descriptive_id
    assert pipeline.descriptive_id == original_pipeline.descriptive_id

    original_opt_graph = deepcopy(opt_graph)
    restored_pipeline = adapter.restore(opt_graph)

    # before change they're equal
    assert opt_graph.descriptive_id == restored_pipeline.descriptive_id

    changed_node = choice(restored_pipeline.nodes)
    changed_node.content['name'] = Operation('ridge')
    changed_node.content['params'].update({'new_hyperparam': 4242})

    # assert that changes to the restored graph don't affect original graph
    assert opt_graph.descriptive_id != restored_pipeline.descriptive_id
    assert opt_graph.descriptive_id == original_opt_graph.descriptive_id


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    return False
    return True
