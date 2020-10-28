import os
import random
from copy import deepcopy
from glob import glob
from math import ceil, log2
from os import remove
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from PIL import Image

from core.composer.chain import Chain, as_nx_graph
from core.utils import default_fedot_data_dir


class ComposerVisualiser:
    default_data_dir = default_fedot_data_dir()
    temp_path = os.path.join(default_data_dir, 'composing_history')
    if 'composing_history' not in os.listdir(default_data_dir):
        os.mkdir(temp_path)
    gif_prefix = 'for_gif_'
    chains_imgs = []
    convergence_imgs = []
    best_chains_imgs = []
    merged_imgs = []

    @staticmethod
    def visualise(chain: Chain, save_path: Optional[str] = None):
        try:
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [4, 1]})
            ComposerVisualiser._visualise(chain, axs[0], 'Current chain')
            ComposerVisualiser._add_table_params(chain, axs[1])
            if not save_path:
                plt.show()
            else:
                plt.savefig(save_path)
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    @staticmethod
    def _visualise(chain: Chain, ax=None, title=None):
        graph, node_labels = as_nx_graph(chain=chain)
        word_labels = list(node_labels.values())
        inv_map = {v: k for k, v in node_labels.items()}
        pos = hierarchy_pos(graph.to_undirected(), root=inv_map[str(chain.root_node)])
        min_size = 3000
        node_sizes = [min_size for _ in word_labels]
        if title:
            plt.title(title)
        colors = colors_by_node_labels(node_labels)
        nx.draw(graph, pos=pos,
                with_labels=False,
                node_size=node_sizes, width=2.0,
                node_color=colors, cmap='Set3', ax=ax)
        for node, (x, y) in pos.items():
            text = '\n'.join(node_labels[node].split('_'))
            if ax is None:
                ax = plt
            ax.text(x, y, text, ha='center', va='center')

    @staticmethod
    def _visualise_chains(chains, fitnesses):
        fitnesses = deepcopy(fitnesses)
        last_best_chain = chains[0]
        prev_fit = fitnesses[0]
        for ch_id, chain in enumerate(chains):
            fig = plt.figure(figsize=(10, 10))
            ComposerVisualiser._visualise(chain, title='Current chain')
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            ComposerVisualiser.chains_imgs.append(img)
            plt.cla()
            plt.clf()
            plt.close('all')
            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_chain = chain
            prev_fit = fitnesses[ch_id]
            fig = plt.figure(figsize=(10, 10))
            ComposerVisualiser._visualise(last_best_chain, title=f'Best chain after {round(ch_id)} evals')
            fig.canvas.draw()
            img = figure_to_array(fig)
            ComposerVisualiser.best_chains_imgs.append(img)
            plt.cla()
            plt.clf()
            plt.close('all')

    @staticmethod
    def _visualise_convergence(fitness_history):
        fitness_history = deepcopy(fitness_history)
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts_set = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {'ts': ts_set, 'fitness': [-f for f in fitness_history]})

        ind = 0
        for ts in ts_set:
            fig = plt.figure()
            plt.rcParams['axes.titlesize'] = 20
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['figure.figsize'] = [10, 10]
            ind += 1
            plt.plot(df['ts'], df['fitness'], label='Composer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)
            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')
            fig.canvas.draw()
            img = figure_to_array(fig)
            ComposerVisualiser.convergence_imgs.append(img)
            plt.cla()
            plt.clf()
            plt.close('all')

    @staticmethod
    def visualise_history(chains, fitnesses):
        print('START VISUALISATION')
        try:
            ComposerVisualiser._clean(with_gif=True)
            ComposerVisualiser._visualise_chains(chains, fitnesses)
            ComposerVisualiser._visualise_convergence(fitnesses)
            ComposerVisualiser._merge_images()
            ComposerVisualiser._combine_gifs()
            ComposerVisualiser._clean()
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    @staticmethod
    def _merge_images():
        for i in range(1, len(ComposerVisualiser.chains_imgs)):
            merged = np.concatenate((ComposerVisualiser.chains_imgs[i],
                                     ComposerVisualiser.best_chains_imgs[i],
                                     ComposerVisualiser.convergence_imgs[i]), axis=1)
            ComposerVisualiser.merged_imgs.append(Image.fromarray(np.uint8(merged)))

    @staticmethod
    def _combine_gifs():
        ComposerVisualiser.merged_imgs[0].save(f'{ComposerVisualiser.temp_path}final_{str(time())}.gif',
                                               save_all=True, append_images=ComposerVisualiser.merged_imgs[1:],
                                               optimize=False, duration=0.5, loop=0)

    @staticmethod
    def _clean(with_gif=False):
        try:
            files = glob(f'{ComposerVisualiser.temp_path}*.png')
            if with_gif:
                files += glob(f'{ComposerVisualiser.temp_path}*.gif')
            for file in files:
                remove(file)
        except Exception as ex:
            print(ex)

    @staticmethod
    def _add_table_params(chain, ax):
        cell_text = []
        columns = ['Model', 'Custom_params']
        for i in chain.nodes:
            lst = [str(i.model.metadata.id)]
            if i.custom_params == 'default_params':
                lst.append('-')
            else:
                lst.append(str(i.custom_params))
            cell_text.append(lst)
        the_table = ax.table(cellText=cell_text,
                             colLabels=columns,
                             loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        ax.axis('off')


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def colors_by_node_labels(node_labels: dict):
    colors = [color for color in range(len(node_labels.keys()))]
    return colors


def scaled_node_size(nodes_amount):
    size = int(7000.0 / ceil(log2(nodes_amount)))
    return size


def hierarchy_pos(graph, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(graph):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(iter(nx.topological_sort(graph)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(graph.nodes))

    def _hierarchy_pos(graph, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(graph.neighbors(root))
        if not isinstance(graph, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(graph, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)
