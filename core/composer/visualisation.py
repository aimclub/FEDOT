import random

import matplotlib.pyplot as plt
import networkx as nx

from core.composer.composer import Chain


class ChainVisualiser:
    def __init__(self):
        pass

    def visualise(self, chain: Chain):
        graph, node_labels = _as_nx_graph(chain=chain)
        root = f'{chain.root_node.node_id}'
        pos = node_positions(graph.to_undirected(), root=root,
                             width=0.5, vert_gap=0.1,
                             vert_loc=0, xcenter=0.5)
        plt.figure(figsize=(10, 16))
        nx.draw(graph, pos=pos,
                with_labels=True, labels=node_labels,
                font_size=12, font_family='calibri', font_weight='bold',
                node_size=7000, width=2.0,
                node_color=colors_by_node_labels(node_labels), cmap='Set3')
        plt.show()


def _as_nx_graph(chain: Chain):
    graph = nx.DiGraph()

    node_labels = {}
    for node in chain.nodes:
        graph.add_node(node.node_id)
        node_labels[node.node_id] = f'{node}'

    def add_edges(graph, chain):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(child.node_id, node.node_id)

    add_edges(graph, chain)
    return graph, node_labels


def colors_by_node_labels(node_labels: dict):
    colors = [color for color in range(len(node_labels.keys()))]
    return colors


def node_positions(G, root=None, width=0.5, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=width, vert_gap=vert_gap,
                       vert_loc=vert_loc, xcenter=xcenter,
                       pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
