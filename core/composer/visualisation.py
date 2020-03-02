import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image

from core.composer.composer import Chain


class ChainVisualiser:
    def __init__(self):
        pass

    def visualise(self, chain: Chain):
        graph, node_labels = _as_nx_graph(chain=chain)
        root = f'{chain.root_node.node_id}'
        pos = node_positions(graph, root=root,
                             width=0.5, vert_gap=0.1,
                             vert_loc=0, xcenter=0.5)
        plt.figure(figsize=(10, 16))
        nx.draw(graph, pos=pos, with_labels=True, labels=node_labels)
        plt.show()


def _as_nx_graph(chain: Chain):
    graph = nx.Graph()

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


class ComposerVisualiser:
    @staticmethod
    def visualise(opt_history):
        fitness_history = [opt_step[1] for opt_step in opt_history]
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {"ts": ts, "fitness": fitness_history})

        images = []
        ind = 0
        for ts in ts:
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 7]

            ind = ind + 1
            ax = plt.subplot()
            ax.plot(df['ts'], df['fitness'], label="Random composer")
            plt.xlabel('Generation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)

            plt.axvline(x=ts, color="black")
            plt.legend(loc="upper left")

            path = f'../../tmp/{ind}.png'
            plt.savefig(path, bbox_inches='tight')
            images.append(Image.open(path))

            plt.cla()
            plt.clf()
            plt.close('all')

            images[0].save(f'../../tmp/conv.gif', save_all=True,
                           append_images=images[1:], duration=250,
                           loop=0)
