from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def simple_linear_pipeline():
    node_first = PrimaryNode('scaling')
    node_isolation_forest = SecondaryNode('isolation_forest_class',
                                          nodes_from=[node_first])
    node_logit = SecondaryNode('logit', [node_first])
    node_final = SecondaryNode('xgboost',
                               nodes_from=[node_isolation_forest, node_logit])
    return Pipeline(node_final)
