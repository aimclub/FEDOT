from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.graph import OptNode


def get_pipeline() -> Pipeline:
    third_level_one = PrimaryNode('lda')

    second_level_one = SecondaryNode('qda', nodes_from=[third_level_one])
    second_level_two = PrimaryNode('qda')

    first_level_one = SecondaryNode('knn', nodes_from=[second_level_one, second_level_two])

    root = SecondaryNode(operation_type='logit', nodes_from=[first_level_one])
    pipeline = Pipeline(root)

    return pipeline


def test_postproc_nodes():
    """
    Test to check if the postproc_nodes method correctly process GraphNodes
    In the process of connecting nodes, GraphNode may appear in the pipeline,
    so the method should change their type to an acceptable one

    postproc_nodes method is called at the end of the update_node method
    """

    pipeline = get_pipeline()

    lda_node = pipeline.nodes[-2]
    qda_node = pipeline.nodes[-1]

    pipeline.connect_nodes(lda_node, qda_node)

    for node in pipeline.nodes:
        assert(isinstance(node, PrimaryNode) or isinstance(node, SecondaryNode))


def test_postproc_opt_nodes():
    """
    Test to check if the postproc_nodes method correctly process OptNodes
    The method should skip OptNodes without changing their type

    postproc_nodes method is called at the end of the update_node method
    """
    pipeline = get_pipeline()

    lda_node = pipeline.nodes[-2]
    qda_node = pipeline.nodes[-1]

    pipeline.connect_nodes(lda_node, qda_node)

    # Check that the postproc_nodes method does not change the type of OptNode type nodes
    new_node = OptNode({'name': "opt"})
    pipeline.update_node(old_node=lda_node,
                         new_node=new_node)
    opt_node = pipeline.nodes[3]
    assert (isinstance(opt_node, OptNode))
