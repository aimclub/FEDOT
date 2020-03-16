import random
from copy import deepcopy

import numpy as np

from core.composer.chain import Chain
from core.composer.gp_composer.gp_node import GPNode
from core.composer.node import NodeGenerator
from core.models.model import LogRegression, KNN, LDA
from core.models.model import XGBoost


def _tree_to_chain(tree_root: GPNode) -> Chain:
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


def test_node_depth_and_high():
    random.seed(1)
    np.random.seed(1)

    last_node = GPNode(chain_node=NodeGenerator.secondary_node(model=XGBoost()))
    last_node_child1 = GPNode(chain_node=NodeGenerator.secondary_node(model=XGBoost()), node_to=last_node)
    last_node_child2 = GPNode(chain_node=NodeGenerator.secondary_node(model=KNN()), node_to=last_node)
    for node in (last_node, last_node_child1, last_node_child2):
        node.nodes_from = []
    for last_node_child in (last_node_child1, last_node_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = GPNode(chain_node=NodeGenerator.primary_node(model=requirement_model, input_data=None),
                              node_to=last_node_child)
            last_node_child.nodes_from.append(new_node)
        last_node.nodes_from.append(last_node_child)

    tree_root_depth = last_node.depth
    tree_secondary_node_depth = last_node.nodes_from[0].depth
    tree_primary_node_depth = last_node.nodes_from[0].nodes_from[0].depth
    tree_secondary_node_height = last_node.nodes_from[0].height
    tree_primary_node_height = last_node.nodes_from[0].nodes_from[0].height
    tree_root_height = last_node.height

    assert not False in [tree_root_depth == 2, tree_secondary_node_depth == 1, tree_primary_node_depth == 0,
                         tree_secondary_node_height == 1, tree_primary_node_height == 2, tree_root_height == 0]
