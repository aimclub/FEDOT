from fedot.core.composer.chain import Chain
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from sensitivity.sensitivity import ChainAnalyze


# def test_one():
#     knn_node = PrimaryNode('knn')
#     lda_node = PrimaryNode('lda')
#     xgb_node = PrimaryNode('xgboost')
#     logit_node = PrimaryNode('logit')
#
#     logit_node_second = SecondaryNode('logit', nodes_from=[knn_node, lda_node])
#     xgb_node_second = SecondaryNode('xgboost', nodes_from=[logit_node])
#
#     qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_second])
#     knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_second, xgb_node])
#
#     knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])
#
#     chain = Chain()
#     chain.add_node(knn_root)
#
#     ChainAnalyze(chain).sample()
