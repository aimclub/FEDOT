from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.optimisers.gp_operators import nodes_from_height
from core.repository.model_types_repository import ModelTypesIdsEnum


def chain_example():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [NodeGenerator.secondary_node(model) for model in (ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.xgboost,
                                                           ModelTypesIdsEnum.knn)]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in (ModelTypesIdsEnum.logit, ModelTypesIdsEnum.lda):
            new_node = NodeGenerator.primary_node(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def test_nodes_from_height():
    chain = chain_example()
    found_nodes = nodes_from_height(chain, 1)
    true_nodes = [node for node in chain.root_node.nodes_from]
    assert all([node_model == found_node for node_model, found_node in zip(true_nodes, found_nodes)])

