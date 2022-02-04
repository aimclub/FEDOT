from copy import deepcopy

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


# ------------------------------------------------------------------------------
# Tests for disconnect_nodes method in GraphOperator

def get_initial_pipeline():
    scaling_node_primary = PrimaryNode('scaling')

    logit_node = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])
    xgb_node = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node, xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_first():
    scaling_node_primary = PrimaryNode('scaling')

    xgb_node_primary = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_second():
    scaling_node_primary = PrimaryNode('scaling')

    xgb_node = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[xgb_node])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_res_pipeline_test_third():
    scaling_node_primary = PrimaryNode('scaling')

    xgb_node = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])
    xgb_node_second = SecondaryNode('xgboost', nodes_from=[scaling_node_primary])

    knn_node_third = SecondaryNode('knn', nodes_from=[xgb_node, xgb_node_second])

    knn_root = SecondaryNode('knn', nodes_from=[knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def test_disconnect_nodes_method_first():
    pipeline = get_initial_pipeline()

    # Disconnect knn_node_third and knn_root nodes
    res_pipeline = get_res_pipeline_test_first()

    knn_node = pipeline.nodes[4]
    knn_root_node = pipeline.nodes[0]

    pipeline.operator.disconnect_nodes(knn_node, knn_root_node)

    assert (res_pipeline == pipeline)


def test_disconnect_nodes_method_second():
    pipeline = get_initial_pipeline()

    # Disconnect xgb_node and knn_node_third
    res_pipeline = get_res_pipeline_test_second()

    xgboost_node = pipeline.nodes[5]
    knn_node = pipeline.nodes[4]

    pipeline.operator.disconnect_nodes(xgboost_node, knn_node)

    assert(res_pipeline == pipeline)


def test_disconnect_nodes_method_third():
    pipeline = get_initial_pipeline()

    # Disconnect qda_node and knn_root nodes
    res_pipeline = get_res_pipeline_test_third()

    qda_node = pipeline.nodes[1]
    knn_root_node = pipeline.nodes[0]

    pipeline.operator.disconnect_nodes(qda_node, knn_root_node)

    assert (res_pipeline == pipeline)


def test_disconnect_nodes_method_fourth():
    pipeline = get_initial_pipeline()

    # Try to disconnect nodes between which there is no edge
    res_pipeline = deepcopy(pipeline)

    xgboost_node = res_pipeline.nodes[2]
    knn_root_node = res_pipeline.nodes[0]

    res_pipeline.operator.disconnect_nodes(xgboost_node, knn_root_node)
    assert (res_pipeline == pipeline)


def test_disconnect_nodes_method_fifth():
    pipeline = get_initial_pipeline()

    # Try to disconnect nodes that are not in this pipeline
    res_pipeline = deepcopy(pipeline)

    xgboost_node = PrimaryNode('xgboost')
    knn_root_node = SecondaryNode('knn', nodes_from=[xgboost_node])

    res_pipeline.operator.disconnect_nodes(xgboost_node, knn_root_node)
    assert (res_pipeline == pipeline)
