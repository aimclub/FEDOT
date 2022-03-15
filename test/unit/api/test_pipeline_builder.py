from typing import Tuple

from fedot.api.api_utils.pipeline_builder import PipelineBuilder, OpT
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode, Node
from fedot.core.pipelines.pipeline import Pipeline


def nodes_same(node_pair: Tuple[Node, Node]) -> bool:
    lhs, rhs = node_pair
    return lhs.descriptive_id == rhs.descriptive_id


def pipelines_same(lhs: Pipeline, rhs: Pipeline) -> bool:
    return all(map(nodes_same, zip(lhs.nodes, rhs.nodes)))


def test_pipeline_builder():

    n1a = PrimaryNode('glm')
    n12a = SecondaryNode('ridge', nodes_from=[n1a])
    n12b = SecondaryNode('glm', nodes_from=[n1a])

    n123a = SecondaryNode('ridge', nodes_from=[n12a, n12b]) # joined
    n123b = SecondaryNode('ridge', nodes_from=[n12b]) # non-joined

    n1b = PrimaryNode('lagged')
    n2b = SecondaryNode('ridge', nodes_from=[n1b])

    # None inputs and empty pipelines
    assert PipelineBuilder().to_pipeline() is None
    assert PipelineBuilder().node(None).to_pipeline() is None
    assert PipelineBuilder().grow_branches(None, None, None).to_pipeline() is None
    assert PipelineBuilder().join('glm').to_pipeline() is None # empty join is empty
    assert PipelineBuilder().join(None).to_pipeline() is None # empty join is empty

    # create builder with prebuilt nodes
    assert pipelines_same( Pipeline(n1a), PipelineBuilder(PrimaryNode('glm')).to_pipeline() ) # single node
    assert pipelines_same( Pipeline(n2b), PipelineBuilder(SecondaryNode('ridge', nodes_from=[PrimaryNode('lagged')])).to_pipeline() ) # single node

    # build single node
    assert pipelines_same( Pipeline(n1a), PipelineBuilder().node('glm').to_pipeline() ) # single node
    assert pipelines_same( Pipeline(n1a), PipelineBuilder().node('glm', branch_idx=42).to_pipeline() ) # index overflow is ok
    assert pipelines_same( Pipeline(n1a), PipelineBuilder().sequence('glm').to_pipeline() ) # single node

    # build linear sequence
    assert pipelines_same( Pipeline(n12a), PipelineBuilder(n1a).node('ridge').to_pipeline() )
    assert pipelines_same( Pipeline(n12a), PipelineBuilder().node('glm').node('ridge').to_pipeline() )
    assert pipelines_same( Pipeline(n12a), PipelineBuilder().sequence('glm', 'ridge').to_pipeline() )
    # following is somewhat an api abuse... but leads to the same sequential pipeline
    assert pipelines_same( Pipeline(n12a), PipelineBuilder().node('glm').branch(None, 'ridge').to_pipeline() )

    # build two parallel sequences
    assert pipelines_same( Pipeline([n1a, n1b]), PipelineBuilder(n1a, n1b).to_pipeline() )
    assert pipelines_same( Pipeline([n1a, n1b]), PipelineBuilder().branch('glm', 'lagged').to_pipeline() )
    assert pipelines_same( Pipeline([n1a, n1b]), PipelineBuilder().grow_branches('glm', 'lagged').to_pipeline() )
    assert pipelines_same( Pipeline([n1a, n1b]), PipelineBuilder().node('glm').node('lagged', branch_idx=1).to_pipeline() )
    assert pipelines_same( Pipeline([n12a, n2b]), PipelineBuilder().branch('glm', 'lagged').grow_branches('ridge', 'ridge').to_pipeline() )

    # empty branch should change nothing
    assert pipelines_same( Pipeline(n1a), PipelineBuilder().node('glm').branch(None, None).to_pipeline() )

    # joined branch
    assert pipelines_same( Pipeline([n12a, n12b]), PipelineBuilder().node('glm').branch('glm', 'ridge').join(None).to_pipeline() ) # no-op join
    assert pipelines_same( Pipeline(n123a), PipelineBuilder().node('glm').branch('glm', 'ridge').join('ridge').to_pipeline() )
