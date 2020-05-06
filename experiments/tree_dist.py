from zss import Node as ZssNode
from zss import simple_distance

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.repository.model_types_repository import ModelTypesIdsEnum


def as_zss_tree(chain):
    root = chain.root_node

    def _as_zss_tree(node):
        zss_node = ZssNode(label=node.descriptive_id)
        if node.nodes_from != None:
            for child in node.nodes_from:
                zss_node.addkid(node=_as_zss_tree(child))
        return zss_node

    zss_root = _as_zss_tree(root)
    return zss_root


first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
second = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.lda)
third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.knn, nodes_from=[first, second])
forth = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost,
                                     nodes_from=[third, forth])

chain = Chain()
for node in [first, second, third, final]:
    chain.add_node(node)

zss_tree = as_zss_tree(chain)

print(simple_distance(zss_tree, zss_tree))
