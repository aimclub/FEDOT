from core.evaluation import LinRegression, LogRegression, XGBoost
from core.node import (
    NodeFactory,
    OperationNode
)


def test_node_log_reg():
    test_node = NodeFactory().log_reg()
    assert test_node.eval_strategy.__class__ == LogRegression


def test_node_lin_log():
    test_node = NodeFactory().lin_reg()
    assert test_node.eval_strategy.__class__ == LinRegression
    assert test_node.__class__ == OperationNode


def test_node_xgboost():
    test_node = NodeFactory().default_xgb()
    assert test_node.eval_strategy.__class__ == XGBoost
