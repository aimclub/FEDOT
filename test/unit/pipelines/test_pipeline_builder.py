from copy import copy

from fedot.core.pipelines.pipeline_builder import PipelineBuilder, merge_pipeline_builders
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from test.unit.dag.test_graph_utils import graphs_same


def builders_same(left: PipelineBuilder, right: PipelineBuilder):
    """ for non-empty builders """
    left_pipeline = left.build()
    right_pipeline = right.build()
    return left_pipeline and right_pipeline and graphs_same(left_pipeline, right_pipeline)


def test_pipeline_builder_with_empty_inputs():
    # None inputs and empty pipelines
    assert PipelineBuilder().build() is None
    assert PipelineBuilder().add_node(None).build() is None
    assert PipelineBuilder().grow_branches(None, None, None).build() is None
    assert PipelineBuilder().join_branches('operation_a').build() is None  # empty join is empty
    assert PipelineBuilder().join_branches(None).build() is None  # empty join is empty


def test_pipeline_builder_with_prebuilt_nodes():
    primary_a = PipelineNode('operation_a')
    primary_b = PipelineNode('operation_b')
    secondary_f = PipelineNode('operation_f', nodes_from=[primary_b])

    # create builder with prebuilt nodes
    assert graphs_same(Pipeline(primary_a), PipelineBuilder(PipelineNode('operation_a')).build())
    assert graphs_same(Pipeline(secondary_f), PipelineBuilder(copy(secondary_f)).build())
    # build two parallel branches
    assert graphs_same(Pipeline([primary_a, primary_b]), PipelineBuilder(primary_a, primary_b).build())


def test_pipeline_builder_with_linear_structure():
    primary_a = PipelineNode('operation_a')
    secondary_f = PipelineNode('operation_f', nodes_from=[primary_a])

    # build single node
    assert graphs_same(Pipeline(primary_a), PipelineBuilder().add_node('operation_a').build())
    assert graphs_same(Pipeline(primary_a), PipelineBuilder().add_sequence('operation_a').build())
    # index overflow is ok
    assert graphs_same(Pipeline(primary_a), PipelineBuilder().add_node('operation_a', branch_idx=42).build())

    # build linear sequence
    assert graphs_same(Pipeline(secondary_f),
                       PipelineBuilder(primary_a).add_node('operation_f').build())
    assert graphs_same(Pipeline(secondary_f),
                       PipelineBuilder().add_node('operation_a').add_node('operation_f').build())
    assert graphs_same(Pipeline(secondary_f),
                       PipelineBuilder().add_sequence('operation_a', 'operation_f').build())

    # empty branch should change nothing
    assert graphs_same(Pipeline(primary_a),
                       PipelineBuilder().add_node('operation_a').add_branch(None, None).build())
    # following is somewhat an api abuse... but leads to the same sequential pipeline
    assert graphs_same(Pipeline(secondary_f),
                       PipelineBuilder().add_node('operation_a').add_branch(None, 'operation_f').build())


def test_pipeline_builder_with_parallel_structure():
    primary_a = PipelineNode('operation_d')
    secondary_f = PipelineNode('operation_f', nodes_from=[primary_a])
    primary_b = PipelineNode('operation_b')
    secondary_h = PipelineNode('operation_h', nodes_from=[primary_b])

    # build two parallel sequences
    assert graphs_same(Pipeline([primary_a, primary_b]),
                       PipelineBuilder().add_branch('operation_d', 'operation_b').build())
    assert graphs_same(Pipeline([primary_a, primary_b]),
                       PipelineBuilder().grow_branches('operation_d', 'operation_b').build())
    assert graphs_same(Pipeline([primary_a, primary_b]),
                       PipelineBuilder().add_node('operation_d').add_node('operation_b', branch_idx=1).build())
    assert graphs_same(Pipeline([secondary_f, secondary_h]),
                       PipelineBuilder().add_branch('operation_d', 'operation_b')
                       .grow_branches('operation_f', 'operation_h').build())


def test_pipeline_builder_with_joined_branches():
    primary_a = PipelineNode('operation_a')
    secondary_f = PipelineNode('operation_f', nodes_from=[primary_a])
    secondary_a = PipelineNode('operation_a', nodes_from=[primary_a])
    joined_f = PipelineNode('operation_f', nodes_from=[secondary_f, secondary_a])  # joined

    # joined branch
    assert graphs_same(Pipeline([secondary_f, secondary_a]),
                       PipelineBuilder().add_node('operation_a')
                       .add_branch('operation_a', 'operation_f')
                       .join_branches(None)
                       .build())
    assert graphs_same(Pipeline(joined_f),
                       PipelineBuilder().add_node('operation_a')
                       .add_branch('operation_a', 'operation_f')
                       .join_branches('operation_f')
                       .build())


def test_pipeline_builder_copy_semantic():
    builder = PipelineBuilder() \
        .add_node('operation_d') \
        .add_node('operation_b') \
        .add_branch('operation_a', 'operation_f') \
        .join_branches('operation_f')

    # repeated builds return same logial results
    assert graphs_same(builder.build(), builder.build())
    # but different objects
    assert id(builder.build()) != id(builder.build())
    assert all([id(n1) != id(n2) for n1, n2 in zip(builder.to_nodes(), builder.to_nodes())])


def test_pipeline_builder_merge_empty():
    builder_one_to_one = PipelineBuilder().add_sequence('operation_a', 'operation_f')

    # empty left, empty right, empty both
    assert builder_one_to_one == merge_pipeline_builders(builder_one_to_one, PipelineBuilder())
    assert builder_one_to_one == merge_pipeline_builders(PipelineBuilder(), builder_one_to_one)
    assert len(merge_pipeline_builders(PipelineBuilder(), PipelineBuilder()).heads) == 0


def test_pipeline_builder_merge_single_node():
    builder_a = PipelineBuilder().add_sequence('operation_a')
    builder_b = PipelineBuilder().add_sequence('operation_f')
    builder_ab = PipelineBuilder().add_sequence('operation_a', 'operation_f')

    assert builders_same(builder_ab, merge_pipeline_builders(builder_a, builder_b))


def test_pipeline_builder_merge_duplicate():
    builder_one_to_one = PipelineBuilder().add_sequence('operation_a', 'operation_f')

    # duplicate builder
    assert builders_same(
        merge_pipeline_builders(builder_one_to_one, builder_one_to_one),
        PipelineBuilder().add_sequence('operation_a', 'operation_f', 'operation_a', 'operation_f')
    )
    # check that builder state is preserved after merge and can be reused
    assert graphs_same(
        builder_one_to_one.build(),
        Pipeline(PipelineNode('operation_f', nodes_from=[PipelineNode('operation_a')]))
    )


def test_pipeline_builder_merge_one_to_one():
    builder_many_to_one = PipelineBuilder().add_branch('operation_b', 'operation_d').join_branches('operation_f')
    builder_one_to_many = PipelineBuilder().add_node('operation_c').add_branch('operation_g', 'operation_a')

    # merge one final to one initial node
    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_one_to_many),
        PipelineBuilder()
        .add_branch('operation_b', 'operation_d').join_branches('operation_f')
        .add_node('operation_c').add_branch('operation_g', 'operation_a')
    )


def test_pipeline_builder_merge_one_to_many():
    builder_many_to_one = PipelineBuilder().add_branch('operation_b', 'operation_d').join_branches('operation_f')
    builder_many_to_many = PipelineBuilder().add_branch('operation_b', 'operation_d').grow_branches('operation_f',
                                                                                                    'operation_h')

    # merge one final to many initial nodes
    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_many_to_many),
        PipelineBuilder()
        .add_branch('operation_b', 'operation_d').join_branches('operation_f')
        .add_branch('operation_b', 'operation_d').grow_branches('operation_f', 'operation_h')
    )


def test_pipeline_builder_merge_many_to_one():
    builder_one_to_one = PipelineBuilder().add_sequence('operation_a', 'operation_f')
    builder_one_to_many = PipelineBuilder().add_node('operation_c').add_branch('operation_g', 'operation_a')

    # merge many final to one initial node
    assert builders_same(
        merge_pipeline_builders(builder_one_to_many, builder_one_to_one),
        PipelineBuilder()
        .add_node('operation_c').add_branch('operation_g', 'operation_a')
        .join_branches('operation_a').add_node('operation_f')
    )


def test_pipeline_builder_merge_many_to_many_undefined():
    builder_one_to_many = PipelineBuilder().add_node('operation_c').add_branch('operation_g', 'operation_a')
    builder_many_to_many = PipelineBuilder().add_branch('operation_b', 'operation_d').grow_branches('operation_f',
                                                                                                    'operation_h')
    # many to many result undefined
    assert merge_pipeline_builders(builder_one_to_many, builder_many_to_many) is None


def test_pipeline_builder_merge_interface():
    builder_one_to_many = PipelineBuilder().add_node('operation_c').add_branch('operation_g', 'operation_a')
    builder_many_to_one = PipelineBuilder().add_branch('operation_b', 'operation_d').join_branches('operation_f')

    assert builders_same(
        merge_pipeline_builders(builder_many_to_one, builder_one_to_many),
        builder_many_to_one.merge_with(builder_one_to_many)
    )


def test_skip_connection_edge():
    pipe_builder = PipelineBuilder() \
        .add_sequence('scaling', 'knn', branch_idx=0) \
        .add_sequence('scaling', 'logit', branch_idx=1) \
        .add_skip_connection_edge(branch_idx_first=0, branch_idx_second=1,
                                  node_idx_in_branch_first=0, node_idx_in_branch_second=1) \
        .join_branches('rf')

    pipeline_with_builder = pipe_builder.build()

    node_scaling_1 = PipelineNode('scaling')
    node_knn = PipelineNode('knn', nodes_from=[node_scaling_1])
    node_scaling_2 = PipelineNode('scaling', nodes_from=[node_knn])
    node_logit = PipelineNode('logit', nodes_from=[node_scaling_2])
    node_rf = PipelineNode('rf', nodes_from=[node_knn, node_logit])

    pipeline_without_builder = Pipeline(node_rf)

    assert graphs_same(pipeline_without_builder, pipeline_with_builder)


def test_skip_connection_edge_to_cycle_graph():
    """ Checks that cycles are avoided even if the edge to cycle graph if manually inserted. """
    pipe = PipelineBuilder()\
        .add_node('operation_b', 0) \
        .add_node('operation_c', 0) \
        .add_node('operation_b2', 1) \
        .add_node('operation_c2', 1)        \
        .join_branches('operation_f')\
        .build()

    pipe_try_cycle = PipelineBuilder()\
        .add_node('operation_b', 0) \
        .add_node('operation_c', 0) \
        .add_node('operation_b2', 1) \
        .add_node('operation_c2', 1)        \
        .join_branches('operation_f')\
        .add_skip_connection_edge(0, 0, 1, 2) \
        .build()

    assert graphs_same(pipe, pipe_try_cycle)
