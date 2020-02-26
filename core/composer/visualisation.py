from enum import Enum

from ete3 import Tree, TreeStyle

from core.composer.composer import Chain
from core.composer.node import Node, PrimaryNode


class ChainStyles(Enum):
    text = 0,
    basic = 1,
    circular = 2,


class ChainVisualiser:
    def __init__(self, style=ChainStyles.basic):
        if style is ChainStyles.circular:
            self.__style = TreeStyle()
            self.__style.mode = "c"
            self.__style.scale = 20
        else:
            raise NotImplementedError()

    def visualise(self, chain: Chain):
        t = ChainTransformer._chain_to_tree(chain)
        t.show(tree_style=self.__style)


class ChainTransformer:
    @staticmethod
    def _chain_to_tree(chain):
        newick_tree = ChainTransformer._node_to_newick(chain.root_node)
        newick_tree = f'({newick_tree});'
        print(newick_tree)
        tree = Tree(newick_tree)
        return tree

    @staticmethod
    def _node_to_newick(node: Node):
        if isinstance(node, PrimaryNode) or node.nodes_from is None:
            return f'{str(type(node.eval_strategy.model))}'
        else:
            parents = []
            for parent_node in node.nodes_from:
                parents.append(f'({ChainTransformer._node_to_newick(parent_node)})')
            return ','.join(parents)
