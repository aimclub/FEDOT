from operators.node import Node, ModelNode, NodeFactory
from operators.evaluation import LinRegression, LogRegression, XGBoost


def test_node_log_reg():
    test_node = NodeFactory().log_reg()
    assert test_node.eval_strategy.__class__ == LogRegression


def test_node_lin_log():
    test_node = NodeFactory().lin_reg()
    assert test_node.eval_strategy.__class__ == LinRegression


def test_node_xgboost():
    test_node = NodeFactory().default_xgb()
    assert test_node.eval_strategy.__class__ == XGBoost
