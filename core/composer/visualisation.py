import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from imageio import get_reader, mimsave

from core.composer.composer import Chain


class ChainVisualiser:
    def __init__(self):
        pass

    @staticmethod
    def visualise(chain: Chain):
        graph, node_labels = _as_nx_graph(chain=chain)
        root = f'{chain.root_node.node_id}'
        pos = node_positions(graph.to_undirected(), root=root,
                             width=0.5, vert_gap=0.1,
                             vert_loc=0, xcenter=0.5)
        plt.figure(figsize=(10, 10))
        nx.draw(graph, pos=pos,
                with_labels=True, labels=node_labels,
                font_size=12, font_family='calibri', font_weight='bold',
                node_size=7000, width=2.0,
                node_color=colors_by_node_labels(node_labels), cmap='Set3')
        plt.show()


    @staticmethod
    def visualise_chains(chains, fitnesses):
        images = []
        images_best = []

        fitnesses = deepcopy(fitnesses)
        last_best_chain = chains[0]

        prev_fit = fitnesses[0]

        for ch_id, chain in enumerate(chains):
            graph, node_labels = _as_nx_graph(chain=deepcopy(chain))
            root = f'{chain.root_node.node_id}'
            pos = node_positions(graph.to_undirected(), root=root,
                                 width=0.5, vert_gap=0.1,
                                 vert_loc=0, xcenter=0.5)
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.title('Current chain')
            nx.draw(graph, pos=pos,
                    with_labels=True, labels=node_labels,
                    font_size=12, font_family='calibri', font_weight='bold',
                    node_size=7000, width=2.0,
                    node_color=colors_by_node_labels(node_labels), cmap='Set3')
            # plt.show()
            path = f'../../tmp/ch_{ch_id}.png'
            plt.savefig(path)
            images.append(Image.open(path))

            plt.cla()
            plt.clf()
            plt.close('all')

            path_best = f'../../tmp/best_ch_{ch_id}.png'

            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_chain = chain
            prev_fit = fitnesses[ch_id]

            best_graph, best_node_labels = _as_nx_graph(chain=deepcopy(last_best_chain))
            best_root = f'{last_best_chain.root_node.node_id}'
            pos = node_positions(best_graph.to_undirected(), root=best_root,
                                 width=0.5, vert_gap=0.1,
                                 vert_loc=0, xcenter=0.5)
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.title(f'Best chain after {round(ch_id)} gens')
            nx.draw(best_graph, pos=pos,
                    with_labels=True, labels=best_node_labels,
                    font_size=12, font_family='calibri', font_weight='bold',
                    node_size=7000, width=2.0,
                    node_color=colors_by_node_labels(best_node_labels), cmap='Set3')

            plt.savefig(path_best)

            plt.cla()
            plt.clf()
            plt.close('all')

            images_best.append(Image.open(path_best))

        images[0].save(f'../../tmp/chains.gif', save_all=True,
                       append_images=images[1:], duration=500,
                       loop=0)

        images_best[0].save(f'../../tmp/chains_best.gif', save_all=True,
                            append_images=images_best[1:], duration=500,
                            loop=0)

    @staticmethod
    def combine_gifs():
        # Create reader object for the gif
        gif0 = get_reader('../../tmp/chains_best.gif')
        gif1 = get_reader('../../tmp/chains.gif')
        gif2 = get_reader('../../tmp/conv.gif')

        # If they don't have the same number of frame take the shorter
        number_of_frames = min(gif0.get_length(), gif1.get_length(), gif2.get_length())

        # Create writer object
        images = []
        for frame_number in range(number_of_frames):
            img0 = gif0.get_next_data()
            img1 = gif1.get_next_data()
            img2 = gif2.get_next_data()
            new_image = np.hstack((img0, img1, img2))
            images.append(new_image)

        gif0.close()
        gif1.close()
        gif2.close()

        mimsave('../../tmp/analyt_full2.gif', images, format='GIF', duration=0.25)


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


class ComposerVisualiser:
    @staticmethod
    def visualise(fitness_history):
        fitness_history = deepcopy(fitness_history)
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {"ts": ts, "fitness": [-f for f in fitness_history]})

        images = []
        ind = 0
        for ts in ts:
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]

            ind = ind + 1
            plt.plot(df['ts'], df['fitness'], label="Random composer")
            plt.xlabel('Generation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)

            plt.axvline(x=ts, color="black")
            plt.legend(loc="upper left")

            path = f'../../tmp/{ind}.png'
            plt.savefig(path)
            images.append(Image.open(path))

            plt.cla()
            plt.clf()
            plt.close('all')

            images[0].save(f'../../tmp/conv.gif', save_all=True,
                           append_images=images[1:], duration=500,
                           loop=0)
