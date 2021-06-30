from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def get_pipeline():
    third_level_one = PrimaryNode('lda')

    second_level_one = SecondaryNode('qda', nodes_from=[third_level_one])
    second_level_two = PrimaryNode('qda')

    first_level_one = SecondaryNode('knn', nodes_from=[second_level_one, second_level_two])

    root = SecondaryNode(operation_type='logit', nodes_from=[first_level_one])
    pipeline = Pipeline(root)

    return pipeline


def test_pipeline_operator_init():
    pipeline = get_pipeline()
    assert type(pipeline.operator) is GraphOperator


def test_distance_to_root_level():
    # given
    pipeline = get_pipeline()
    selected_node = pipeline.nodes[2]

    # when
    height = pipeline.operator.distance_to_root_level(selected_node)

    # then
    assert height == 2


def test_nodes_from_layer():
    # given
    pipeline = get_pipeline()
    desired_layer = 2

    # when
    nodes_from_desired_layer = pipeline.operator.nodes_from_layer(desired_layer)

    # then
    assert len(nodes_from_desired_layer) == 2


def test_actualise_old_node_children():
    # given
    pipeline = get_pipeline()
    selected_node = pipeline.nodes[2]
    new_node = PrimaryNode('knnreg')

    # when
    pipeline.operator.actualise_old_node_children(old_node=selected_node,
                                                  new_node=new_node)
    updated_parent = pipeline.nodes[1]

    # then
    assert new_node in updated_parent.nodes_from


def test_sort_nodes():
    # given
    pipeline = get_pipeline()
    selected_node = pipeline.nodes[2]
    original_length = pipeline.length
    new_node = PrimaryNode('knnreg')
    new_subroot = SecondaryNode('knnreg', nodes_from=[new_node])

    # when
    selected_node.nodes_from.append(new_subroot)
    pipeline.operator.sort_nodes()

    # then
    assert pipeline.length == original_length + 2
    assert pipeline.nodes[4] is new_subroot
    assert pipeline.nodes[5] is new_node


def test_node_children():
    # given
    pipeline = get_pipeline()
    selected_node = pipeline.nodes[2]

    # when
    children = pipeline.operator.node_children(selected_node)

    # then
    assert len(children) == 1
    assert children[0] is pipeline.nodes[1]
