from operators.node import Node, ModelNode, NodeFactory
from operators.evaluation import LinRegression, LogRegression
from typing import List


def test_model_node_log_reg():
    model_node = ModelNode(nodes_from=List['Node'],
                           nodes_to=List['Node'],
                           data_stream=[1, 2, 3],
                           )
    test_model = NodeFactory(model_node).log_reg()
    assert test_model.evaluation_strategy.evaluate() == 'LogRegPredict'
    assert model_node.evaluation_strategy.__class__ == LogRegression


def test_model_node_lin_log():
    model_node = ModelNode(nodes_from=List['Node'],
                           nodes_to=List['Node'],
                           data_stream=[1, 2, 3],
                           )
    test_model = NodeFactory(model_node).lin_reg()
    assert test_model.evaluation_strategy.evaluate() == 'LinRegPredict'
    assert model_node.evaluation_strategy.__class__ == LinRegression
