from typing import Tuple

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode, Node
from fedot.core.pipelines.pipeline import Pipeline


def nodes_same(node_pair: Tuple[Node, Node]) -> bool:
    lhs, rhs = node_pair
    return lhs.descriptive_id == rhs.descriptive_id


def pipelines_same(lhs: Pipeline, rhs: Pipeline) -> bool:
    return all(map(nodes_same, zip(lhs.nodes, rhs.nodes)))


def test_pipeline_builder_with_empty_inputs():
    # None inputs and empty pipelines
    assert PipelineBuilder().to_pipeline() is None
    assert PipelineBuilder().add_node(None).to_pipeline() is None
    assert PipelineBuilder().grow_branches(None, None, None).to_pipeline() is None
    assert PipelineBuilder().join_branches('glm').to_pipeline() is None  # empty join is empty
    assert PipelineBuilder().join_branches(None).to_pipeline() is None  # empty join is empty


def test_pipeline_builder_with_prebuilt_nodes():
    primary_a = PrimaryNode('glm')
    primary_b = PrimaryNode('lagged')
    secondary_ba = SecondaryNode('ridge', nodes_from=[primary_b])

    # create builder with prebuilt nodes
    assert pipelines_same(Pipeline(primary_a), PipelineBuilder(PrimaryNode('glm')).to_pipeline())
    assert pipelines_same(Pipeline(secondary_ba),
                          PipelineBuilder(SecondaryNode('ridge', nodes_from=[PrimaryNode('lagged')])).to_pipeline())
    # build two parallel branches
    assert pipelines_same(Pipeline([primary_a, primary_b]), PipelineBuilder(primary_a, primary_b).to_pipeline())


def test_pipeline_builder_with_linear_structure():
    primary_a = PrimaryNode('glm')
    secondary_aa = SecondaryNode('ridge', nodes_from=[primary_a])

    # build single node
    assert pipelines_same(Pipeline(primary_a), PipelineBuilder().add_node('glm').to_pipeline())  # single node
    assert pipelines_same(Pipeline(primary_a),
                          PipelineBuilder().add_node('glm', branch_idx=42).to_pipeline())  # index overflow is ok
    assert pipelines_same(Pipeline(primary_a), PipelineBuilder().add_sequence('glm').to_pipeline())  # single node

    # build linear sequence
    assert pipelines_same(Pipeline(secondary_aa), PipelineBuilder(primary_a).add_node('ridge').to_pipeline())
    assert pipelines_same(Pipeline(secondary_aa), PipelineBuilder().add_node('glm').add_node('ridge').to_pipeline())
    assert pipelines_same(Pipeline(secondary_aa), PipelineBuilder().add_sequence('glm', 'ridge').to_pipeline())

    # empty branch should change nothing
    assert pipelines_same(Pipeline(primary_a),
                          PipelineBuilder().add_node('glm').add_branch(None, None).to_pipeline())
    # following is somewhat an api abuse... but leads to the same sequential pipeline
    assert pipelines_same(Pipeline(secondary_aa),
                          PipelineBuilder().add_node('glm').add_branch(None, 'ridge').to_pipeline())


def test_pipeline_builder_with_parallel_structure():
    primary_a = PrimaryNode('glm')
    secondary_aa = SecondaryNode('ridge', nodes_from=[primary_a])
    primary_b = PrimaryNode('lagged')
    secondary_ba = SecondaryNode('ridge', nodes_from=[primary_b])

    # build two parallel sequences
    assert pipelines_same(Pipeline([primary_a, primary_b]), PipelineBuilder().add_branch('glm', 'lagged').to_pipeline())
    assert pipelines_same(Pipeline([primary_a, primary_b]),
                          PipelineBuilder().grow_branches('glm', 'lagged').to_pipeline())
    assert pipelines_same(Pipeline([primary_a, primary_b]),
                          PipelineBuilder().add_node('glm').add_node('lagged', branch_idx=1).to_pipeline())
    assert pipelines_same(Pipeline([secondary_aa, secondary_ba]),
                          PipelineBuilder().add_branch('glm', 'lagged').grow_branches('ridge', 'ridge').to_pipeline())


def test_pipeline_builder_with_joined_branches():
    primary_a = PrimaryNode('glm')
    secondary_aa = SecondaryNode('ridge', nodes_from=[primary_a])
    secondary_ab = SecondaryNode('glm', nodes_from=[primary_a])
    joined_a = SecondaryNode('ridge', nodes_from=[secondary_aa, secondary_ab])  # joined

    # joined branch
    assert pipelines_same(Pipeline([secondary_aa, secondary_ab]),
                          PipelineBuilder()
                          .add_node('glm').add_branch('glm', 'ridge').join_branches(None).to_pipeline())
    assert pipelines_same(Pipeline(joined_a),
                          PipelineBuilder()
                          .add_node('glm').add_branch('glm', 'ridge').join_branches('ridge').to_pipeline())
