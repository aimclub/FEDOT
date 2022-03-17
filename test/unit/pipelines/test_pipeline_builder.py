from typing import Tuple

from fedot.core.pipelines.pipeline_builder import PipelineBuilder, merge_pipeline_builders
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode, Node
from fedot.core.pipelines.pipeline import Pipeline


def nodes_same(node_pair: Tuple[Node, Node]) -> bool:
    left, right = node_pair
    return left.descriptive_id == right.descriptive_id


def pipelines_same(left: Pipeline, right: Pipeline) -> bool:
    # return GraphOperator(left).is_graph_equal(right)
    left_set = set(map(lambda n: n.descriptive_id, left.nodes))
    right_set = set(map(lambda n: n.descriptive_id, right.nodes))
    return left_set == right_set


def builders_same(left: PipelineBuilder, right: PipelineBuilder):
    """ for non-empty builders """
    left_pipeline = left.to_pipeline()
    right_pipeline = right.to_pipeline()
    return left_pipeline and right_pipeline and pipelines_same(left_pipeline, right_pipeline)


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
    secondary_ba = SecondaryNode('ar', nodes_from=[primary_b])

    # build two parallel sequences
    assert pipelines_same(Pipeline([primary_a, primary_b]), PipelineBuilder().add_branch('glm', 'lagged').to_pipeline())
    assert pipelines_same(Pipeline([primary_a, primary_b]),
                          PipelineBuilder().grow_branches('glm', 'lagged').to_pipeline())
    assert pipelines_same(Pipeline([primary_a, primary_b]),
                          PipelineBuilder().add_node('glm').add_node('lagged', branch_idx=1).to_pipeline())
    assert pipelines_same(Pipeline([secondary_aa, secondary_ba]),
                          PipelineBuilder().add_branch('glm', 'lagged').grow_branches('ridge', 'ar').to_pipeline())


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


def test_pipeline_builder_copy_semantic():
    builder = PipelineBuilder().add_node('glm').add_node('lagged').add_branch('glm', 'ridge').join_branches('ridge')

    # repeated builds return same logial results
    assert pipelines_same(builder.to_pipeline(), builder.to_pipeline())
    # but different objects
    assert id(builder.to_pipeline()) != id(builder.to_pipeline())
    assert all([id(n1) != id(n2) for n1, n2 in zip(builder.to_nodes(), builder.to_nodes())])


def test_pipeline_builder_merge_empty():

    builder_one_to_one = PipelineBuilder().add_sequence('glm', 'ridge')

    # empty left, empty right, empty both
    assert builder_one_to_one == merge_pipeline_builders(builder_one_to_one, PipelineBuilder())
    assert builder_one_to_one == merge_pipeline_builders(PipelineBuilder(), builder_one_to_one)
    assert len(merge_pipeline_builders(PipelineBuilder(), PipelineBuilder()).heads) == 0


def test_pipeline_builder_merge_duplicate():

    builder_one_to_one = PipelineBuilder().add_sequence('glm', 'ridge')

    # duplicate builder
    assert builders_same(
        merge_pipeline_builders(builder_one_to_one, builder_one_to_one),
        PipelineBuilder().add_sequence('glm', 'ridge', 'glm', 'ridge')
    )
    # check that builder state is preserved after merge and can be reused
    assert pipelines_same(
        builder_one_to_one.to_pipeline(),
        Pipeline(SecondaryNode('ridge', nodes_from=[PrimaryNode('glm')]))
    )


def test_pipeline_builder_merge_one_to_one():

    builder_many_to_one = PipelineBuilder().add_branch('lagged', 'smoothing').join_branches('ridge')
    builder_one_to_many = PipelineBuilder().add_node('scaling').add_branch('rf', 'glm')

    # merge one final to one initial node
    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_one_to_many),
        PipelineBuilder().add_branch('lagged', 'smoothing').join_branches('ridge')
                         .add_node('scaling').add_branch('rf', 'glm')
    )


def test_pipeline_builder_merge_one_to_many():

    builder_many_to_one = PipelineBuilder().add_branch('lagged', 'smoothing').join_branches('ridge')
    builder_many_to_many = PipelineBuilder().add_branch('lagged', 'smoothing').grow_branches('ridge', 'ar')

    # merge one final to many initial nodes
    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_many_to_many),
        PipelineBuilder().add_branch('lagged', 'smoothing').join_branches('ridge')
                         .add_branch('lagged', 'smoothing').grow_branches('ridge', 'ar')
    )


def test_pipeline_builder_merge_many_to_one():

    builder_one_to_one = PipelineBuilder().add_sequence('glm', 'ridge')
    builder_one_to_many = PipelineBuilder().add_node('scaling').add_branch('rf', 'glm')

    # merge many final to one initial node
    assert builders_same(
        merge_pipeline_builders(builder_one_to_many, builder_one_to_one),
        PipelineBuilder().add_node('scaling').add_branch('rf', 'glm')
                         .join_branches('glm').add_node('ridge')
    )


def test_pipeline_builder_merge_many_to_many():

    builder_one_to_many = PipelineBuilder().add_node('scaling').add_branch('rf', 'glm')
    builder_many_to_many = PipelineBuilder().add_branch('lagged', 'smoothing').grow_branches('ridge', 'ar')

    # many to many result undefined
    assert merge_pipeline_builders(builder_one_to_many, builder_many_to_many) is None


def test_pipeline_builder_merge_interface():

    builder_one_to_many = PipelineBuilder().add_node('scaling').add_branch('rf', 'glm')
    builder_many_to_one = PipelineBuilder().add_branch('lagged', 'smoothing').join_branches('ridge')

    # check interface method
    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_one_to_many),
        builder_many_to_one.merge_with(builder_one_to_many)
    )