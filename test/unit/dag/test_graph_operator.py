from copy import deepcopy

from fedot.core.dag.graph_operator import GraphOperator, get_distance_between
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.graph import OptNode
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def get_pipeline() -> Pipeline:
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
    height = pipeline.distance_to_root_level(selected_node)

    # then
    assert height == 2


def test_nodes_from_layer():
    # given
    pipeline = get_pipeline()
    desired_layer = 2

    # when
    nodes_from_desired_layer = pipeline.nodes_from_layer(desired_layer)

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
    children = pipeline.node_children(selected_node)

    # then
    assert len(children) == 1
    assert children[0] is pipeline.nodes[1]


# ------------------------------------------------------------------------------
# Tests for distance_to_other method

def test_distance_to_same_pipeline_restored():
    # given
    adapter = PipelineAdapter()
    pipeline = get_pipeline()
    opt_graph = adapter.adapt(pipeline)

    # when
    distance = get_distance_between(graph_1=pipeline, graph_2=adapter.restore(opt_graph))

    # then
    assert distance == 0


def test_known_distances():
    pipeline_scaling = PipelineBuilder().add_node('scaling').to_pipeline()  # scaling
    pipeline_xgboost = PipelineBuilder().add_node('xgboost').to_pipeline()  # xgboost
    pipeline_knn = PipelineBuilder().add_node('scaling').add_node('knn').to_pipeline()  # scaling -> knn
    pipeline_linear = PipelineBuilder().add_node('scaling').add_node('linear').to_pipeline()  # scaling -> linear
    pipeline_knn_alternate_params = PipelineBuilder().add_node('scaling').\
        add_node('knn', params={'metric': 'euclidean'}).to_pipeline()  # scaling -> knn_alternate_params

    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_knn) == 0  # the same pipeline
    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_scaling) == 2  # changes: 1 node (operation) + 1 edge
    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_linear) == 1  # changes: 1 node (operation)
    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_knn_alternate_params) == 1  # changes: 1 node (params)
    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_xgboost) == 3  # changes: 2 nodes (operations) + 1 edge
    assert get_distance_between(graph_1=pipeline_knn,
                                graph_2=pipeline_knn_alternate_params) == 1  # changes: 1 operation + params


# ------------------------------------------------------------------------------
# Tests for disconnect_nodes method

def get_initial_pipeline():
    scaling_node_primary = PrimaryNode('scaling')

    logit_node = SecondaryNode('rf', nodes_from=[scaling_node_primary])
    rf_node = SecondaryNode('rf', nodes_from=[scaling_node_primary])
    rf_node_second = SecondaryNode('rf', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[rf_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node, rf_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_first():
    scaling_node_primary = PrimaryNode('scaling')

    rf_node_primary = SecondaryNode('rf', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[rf_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_second():
    scaling_node_primary = PrimaryNode('scaling')

    rf_node = SecondaryNode('rf', nodes_from=[scaling_node_primary])
    rf_node_second = SecondaryNode('rf', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[rf_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[rf_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_third():
    scaling_node_primary = PrimaryNode('scaling')

    rf_node = SecondaryNode('rf', nodes_from=[scaling_node_primary])
    rf_node_second = SecondaryNode('rf', nodes_from=[scaling_node_primary])

    knn_node_third = SecondaryNode('knn', nodes_from=[rf_node, rf_node_second])

    knn_root = SecondaryNode('knn', nodes_from=[knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def test_disconnect_nodes_method_first():
    pipeline = get_initial_pipeline()

    # Disconnect knn_node_third and knn_root nodes
    res_pipeline = get_res_pipeline_test_first()

    knn_node = pipeline.nodes[4]
    knn_root_node = pipeline.nodes[0]

    pipeline.disconnect_nodes(knn_node, knn_root_node)

    assert res_pipeline == pipeline


def test_disconnect_nodes_method_second():
    pipeline = get_initial_pipeline()

    # Disconnect xgb_node and knn_node_third
    res_pipeline = get_res_pipeline_test_second()

    rf_node = pipeline.nodes[5]
    knn_node = pipeline.nodes[4]

    pipeline.disconnect_nodes(rf_node, knn_node)

    assert res_pipeline == pipeline


def test_disconnect_nodes_method_third():
    pipeline = get_initial_pipeline()

    # Disconnect qda_node and knn_root nodes
    res_pipeline = get_res_pipeline_test_third()

    qda_node = pipeline.nodes[1]
    knn_root_node = pipeline.nodes[0]

    pipeline.disconnect_nodes(qda_node, knn_root_node)

    assert res_pipeline == pipeline


def test_disconnect_nodes_method_fourth():
    pipeline = get_initial_pipeline()

    # Try to disconnect nodes between which there is no edge
    res_pipeline = deepcopy(pipeline)

    rf_node = res_pipeline.nodes[2]
    knn_root_node = res_pipeline.nodes[0]

    res_pipeline.disconnect_nodes(rf_node, knn_root_node)
    assert res_pipeline == pipeline


def test_disconnect_nodes_method_fifth():
    pipeline = get_initial_pipeline()

    # Try to disconnect nodes that are not in this pipeline
    res_pipeline = deepcopy(pipeline)

    rf_node = PrimaryNode('rf')
    knn_root_node = SecondaryNode('knn', nodes_from=[rf_node])

    res_pipeline.disconnect_nodes(rf_node, knn_root_node)
    assert res_pipeline == pipeline


# ------------------------------------------------------------------------------
# Test for get_edges method

def test_get_edges():
    pipeline = get_pipeline()

    lda = pipeline.nodes[3]
    qda_second = pipeline.nodes[2]
    qda_first = pipeline.nodes[4]
    knn = pipeline.nodes[1]
    logit = pipeline.nodes[0]

    res_edges = [(knn, logit), (qda_second, knn), (qda_first, knn), (lda, qda_second)]

    edges = pipeline.get_edges()
    assert res_edges == edges


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
