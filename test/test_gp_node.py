from copy import deepcopy

from core.composer.chain import Chain
from core.composer.gp_composer.gp_node import GPNode
from core.composer.gp_composer.gp_node import swap_nodes
from core.composer.node import NodeGenerator
from core.models.model import LogRegression, KNN, LDA, MLP
from core.models.model import XGBoost


def tree_to_chain(tree_root: GPNode) -> Chain:
    chain = Chain()
    nodes = flat_nodes_tree(deepcopy(tree_root))
    for node in nodes:
        if node.nodes_from:
            for i in range(len(node.nodes_from)):
                node.nodes_from[i] = node.nodes_from[i].chain_node
        chain.add_node(node.chain_node)
    return chain


def flat_nodes_tree(node):
    if node.nodes_from:
        nodes = []
        for children in node.nodes_from:
            nodes += flat_nodes_tree(children)
        return [node] + nodes
    else:
        return [node]


def tree1():
    root_of_tree1 = GPNode(chain_node=NodeGenerator.secondary_node(model=XGBoost()))
    root_child1 = GPNode(chain_node=NodeGenerator.secondary_node(model=XGBoost()), node_to=root_of_tree1)
    root_child2 = GPNode(chain_node=NodeGenerator.secondary_node(model=MLP()), node_to=root_of_tree1)
    for node in (root_of_tree1, root_child1, root_child2):
        node.nodes_from = []
    for last_node_child in (root_child1, root_child2):
        for requirement_model in (KNN(), LDA()):
            new_node = GPNode(chain_node=NodeGenerator.primary_node(model=requirement_model, input_data=None),
                              node_to=last_node_child)
            last_node_child.nodes_from.append(new_node)
        root_of_tree1.nodes_from.append(last_node_child)
    return root_of_tree1


def tree2():
    root_of_tree2 = GPNode(chain_node=NodeGenerator.secondary_node(XGBoost()))
    root_child1 = GPNode(chain_node=NodeGenerator.secondary_node(XGBoost()), node_to=root_of_tree2)
    root_child2 = GPNode(chain_node=NodeGenerator.secondary_node(KNN()), node_to=root_of_tree2)
    for node in (root_of_tree2, root_child1, root_child2):
        node.nodes_from = []
    new_node = GPNode(chain_node=NodeGenerator.primary_node(LogRegression(), input_data=None), node_to=root_child1)
    root_child1.nodes_from.append(new_node)
    new_node = GPNode(NodeGenerator.secondary_node(XGBoost()), node_to=root_child1)
    new_node.nodes_from = []
    new_node.nodes_from.append(GPNode(NodeGenerator.primary_node(KNN(), input_data=None), node_to=new_node))
    new_node.nodes_from.append(GPNode(NodeGenerator.primary_node(LDA(), input_data=None), node_to=new_node))
    root_child1.nodes_from.append(new_node)
    root_of_tree2.nodes_from.append(root_child1)
    root_child2.nodes_from.append(
        GPNode(NodeGenerator.primary_node(LogRegression(), input_data=None), node_to=root_child2))
    root_child2.nodes_from.append(GPNode(NodeGenerator.primary_node(LDA(), input_data=None), node_to=root_child2))
    root_of_tree2.nodes_from.append(root_child2)
    return root_of_tree2


def test_node_depth_and_height():
    last_node = tree1()

    tree_root_depth = last_node.depth
    tree_secondary_node_depth = last_node.nodes_from[0].depth
    tree_primary_node_depth = last_node.nodes_from[0].nodes_from[0].depth

    assert all([tree_root_depth == 2, tree_secondary_node_depth == 1, tree_primary_node_depth == 0])

    tree_secondary_node_height = last_node.nodes_from[0].height
    tree_primary_node_height = last_node.nodes_from[0].nodes_from[0].height
    tree_root_height = last_node.height

    assert all([tree_secondary_node_height == 1, tree_primary_node_height == 2, tree_root_height == 0])


def test_swap_nodes():
    root_of_tree1 = tree1()

    root_of_tree2 = tree2()

    height_in_tree1 = 1
    height_in_tree2 = 1

    # nodes_from_height function check
    nodes_set_tree1 = root_of_tree1.nodes_from_height(height_in_tree1)
    assert len(nodes_set_tree1) == 2
    nodes_set_tree2 = root_of_tree2.nodes_from_height(height_in_tree2)
    assert len(nodes_set_tree2) == 2

    tree1_node = nodes_set_tree1[1]
    tree2_node = nodes_set_tree2[0]

    assert isinstance(tree1_node.eval_strategy.model, MLP)
    assert isinstance(tree2_node.eval_strategy.model, XGBoost)
    swap_nodes(tree1_node, tree2_node)

    chain = tree_to_chain(root_of_tree1)
    assert len(chain.nodes) == 9

    correct_nodes = [XGBoost, XGBoost, KNN, LDA, XGBoost, LogRegression, XGBoost, KNN, LDA]
    # swap_nodes function check
    assert all([isinstance(model_after_swap.eval_strategy.model, correct_model) for model_after_swap, correct_model in
                zip(chain.nodes, correct_nodes)])
