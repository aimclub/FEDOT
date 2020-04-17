from copy import deepcopy

from core.composer.chain import Chain
from core.composer.optimisers.gp_node import GPNode
from core.composer.node import SecondaryNode
from core.composer.optimisers.gp_node import swap_nodes
from core.composer.node import NodeGenerator
from core.repository.model_types_repository import ModelTypesIdsEnum


def tree_to_chain(tree_root: GPNode) -> Chain:
    chain = Chain()
    nodes = flat_nodes_tree(deepcopy(tree_root))
    for node in nodes:
        if node.nodes_from:
            for i in range(len(node.nodes_from)):
                node.nodes_from[i] = node.nodes_from[i].chain_node
        chain.add_node(node.chain_node)
    return chain


def chain_to_tree(chain: Chain) -> GPNode:
    chain = deepcopy(chain)
    def recursive_nodes_to_tree(prev_node: GPNode):
        for i, node_parent in enumerate(prev_node.nodes_from):
            root_of_tree = GPNode(chain_node=node_parent, node_to=prev_node)
            prev_node.nodes_from[i] = root_of_tree
            if isinstance(root_of_tree.chain_node, SecondaryNode):
                recursive_nodes_to_tree(root_of_tree)

    root = GPNode(chain_node=chain.root_node)
    if isinstance(root.chain_node, SecondaryNode):
        recursive_nodes_to_tree(root)
    return root


def flat_nodes_tree(node):
    if node.nodes_from:
        nodes = []
        for children in node.nodes_from:
            nodes += flat_nodes_tree(children)
        return [node] + nodes
    else:
        return [node]


def tree_first():
    root_of_tree = GPNode(chain_node=NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.xgboost))
    root_child_first, root_child_second = [GPNode(chain_node=NodeGenerator.secondary_node(model_type=model),
                                                  node_to=root_of_tree) for model in
                                           (ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.mlp)]
    for last_node_child in (root_child_first, root_child_second):
        for requirement_model in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda):
            new_node = GPNode(chain_node=NodeGenerator.primary_node(model_type=requirement_model),
                              node_to=last_node_child)
            last_node_child.nodes_from.append(new_node)
        root_of_tree.nodes_from.append(last_node_child)
    return root_of_tree


def tree_second():
    root_of_tree = GPNode(chain_node=NodeGenerator.secondary_node(ModelTypesIdsEnum.xgboost))
    root_child_first, root_child_second = [
        GPNode(chain_node=NodeGenerator.secondary_node(model_type=model), node_to=root_of_tree) for model in
        (ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn)]

    new_node = GPNode(chain_node=NodeGenerator.primary_node(ModelTypesIdsEnum.logit), node_to=root_child_first)
    root_child_first.nodes_from.append(new_node)

    new_node = GPNode(NodeGenerator.secondary_node(ModelTypesIdsEnum.xgboost), node_to=root_child_first)

    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda):
        new_node.nodes_from.append(GPNode(NodeGenerator.primary_node(model_type), node_to=new_node))

    root_child_first.nodes_from.append(new_node)
    root_of_tree.nodes_from.append(root_child_first)

    for model_type in (ModelTypesIdsEnum.logit, ModelTypesIdsEnum.lda):
        root_child_second.nodes_from.append(
            GPNode(NodeGenerator.primary_node(model_type), node_to=root_child_second))

    root_of_tree.nodes_from.append(root_child_second)
    return root_of_tree

def chain_first():
    #    XG
    #  |     \
    # XG      KNN
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

def test_node_depth_and_height():
    last_node = tree_first()

    tree_root_depth = last_node.depth
    tree_secondary_node_depth = last_node.nodes_from[0].depth
    tree_primary_node_depth = last_node.nodes_from[0].nodes_from[0].depth

    assert all([tree_root_depth == 2, tree_secondary_node_depth == 1, tree_primary_node_depth == 0])

    tree_secondary_node_height = last_node.nodes_from[0].height
    tree_primary_node_height = last_node.nodes_from[0].nodes_from[0].height
    tree_root_height = last_node.height

    assert all([tree_secondary_node_height == 1, tree_primary_node_height == 2, tree_root_height == 0])


def test_swap_nodes():
    root_of_tree_first = tree_first()

    root_of_tree_second = tree_second()

    height_in_tree = 1

    # nodes_from_height function check
    nodes_set_tree_first = root_of_tree_first.nodes_from_height(height_in_tree)
    assert len(nodes_set_tree_first) == 2
    nodes_set_tree_second = root_of_tree_second.nodes_from_height(height_in_tree)
    assert len(nodes_set_tree_second) == 2

    tree_first_node = nodes_set_tree_first[1]
    tree_second_node = nodes_set_tree_second[0]

    assert tree_first_node.chain_node.model.model_type == ModelTypesIdsEnum.mlp
    assert tree_second_node.chain_node.model.model_type, ModelTypesIdsEnum.xgboost

    swap_nodes(tree_first_node, tree_second_node)

    chain = tree_to_chain(root_of_tree_first)
    assert len(chain.nodes) == 9

    correct_nodes = [ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.xgboost,
                     ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda, ModelTypesIdsEnum.xgboost,
                     ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost,
                     ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda]
    # swap_nodes function check
    assert all([model_after_swap.model.model_type == correct_model for model_after_swap, correct_model in
                zip(chain.nodes, correct_nodes)])

def test_chain_to_tree():
    chain = chain_first()
    gp_root_of_chain = chain_to_tree(chain)
    assert tree_to_chain(gp_root_of_chain) == chain

