from copy import deepcopy
from random import choice

import pytest
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptNode

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def get_pipelines():
    one_node_pipeline = PipelineBuilder() \
        .add_sequence('logit') \
        .build()
    linear_pipeline = PipelineBuilder() \
        .add_sequence('logit', 'logit', 'logit') \
        .build()
    branching_structure = PipelineBuilder() \
        .add_node('operation_a') \
        .add_branch('operation_a', 'operation_f') \
        .join_branches('operation_f') \
        .build()
    branching_structure2 = PipelineBuilder() \
        .add_node('operation_a') \
        .add_branch('operation_b', 'operation_c') \
        .grow_branches('operation_d', None) \
        .join_branches('operation_f') \
        .add_node('operation_a') \
        .build()
    node_a = PipelineNode('logit')
    node_b = PipelineNode('logit', nodes_from=[node_a])
    node_c = PipelineNode('logit', nodes_from=[node_b, node_a])
    skip_connection_structure = Pipeline(node_c)

    return [one_node_pipeline, linear_pipeline,
            branching_structure, branching_structure2,
            skip_connection_structure]


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_pipelines_adapt_properly(pipeline):
    adapter = PipelineAdapter()
    verifier = GraphVerifier(DEFAULT_DAG_RULES)

    assert all(isinstance(node, PipelineNode) for node in pipeline.nodes)
    assert _check_nodes_references_correct(pipeline)
    assert verifier(pipeline)

    opt_graph = adapter.adapt(pipeline)

    assert all(type(node) is OptNode for node in opt_graph.nodes)  # checking strict type equality!
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
    changed_node.content['name'] = 'another_operation'
    changed_node.content['params'].update({'new_hyperparam': 39})

    # assert that changes to the adapted pipeline don't affect original graph
    assert pipeline.descriptive_id != opt_graph.descriptive_id
    assert pipeline.descriptive_id == original_pipeline.descriptive_id

    original_opt_graph = deepcopy(opt_graph)
    restored_pipeline = adapter.restore(opt_graph)

    # before change they're equal
    assert opt_graph.descriptive_id == restored_pipeline.descriptive_id

    changed_node = choice(restored_pipeline.nodes)
    changed_node.content['name'] = 'yet_another_operation'
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
