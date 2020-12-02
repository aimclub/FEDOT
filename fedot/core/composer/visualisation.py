import os
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

from fedot.core.chains.chain import Chain, as_nx_graph
from fedot.core.utils import default_fedot_data_dir


class ChainVisualiser:

    def __init__(self):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        if 'composing_history' not in os.listdir(default_data_dir):
            os.mkdir(self.temp_path)

        self.chains_imgs = []
        self.convergence_imgs = []
        self.best_chains_imgs = []
        self.merged_imgs = []

    def visualise(self, chain: Chain, save_path: Optional[str] = None, params_save_path: Optional[str] = None):
        try:
            fig, axs = plt.subplots(figsize=(9, 9))
            fig.suptitle('Current chain')
            self._visualise_chain(chain, axs)
            if params_save_path:
                self._save_chain_params(chain, params_save_path)
            if not save_path:
                plt.show()
            else:
                plt.savefig(save_path)
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    def _visualise_chain(self, chain: Chain, ax=None, title=None):
        pos, node_labels = self._draw_tree(chain, ax, title)
        self._draw_labels(pos, node_labels, ax)

    def _draw_tree(self, chain: Chain, ax=None, title=None):
        graph, node_labels = as_nx_graph(chain=chain)
        word_labels = list(node_labels.values())
        inv_map = {v: k for k, v in node_labels.items()}
        pos = hierarchy_pos(graph.to_undirected(), root=inv_map[str(chain.root_node)])
        min_size = 3000
        node_sizes = [min_size for _ in word_labels]
        if title:
            plt.title(title)
        colors = colors_by_node_labels(node_labels)
        nx.draw(graph, pos=pos, with_labels=False,
                node_size=node_sizes, width=2.0,
                node_color=colors, cmap='Set3', ax=ax)
        return pos, node_labels

    def _draw_labels(self, pos, node_labels, ax):
        for node, (x, y) in pos.items():
            text = '\n'.join(node_labels[node].split('_'))
            if ax is None:
                ax = plt
            ax.text(x, y, text, ha='center', va='center')

    def _visualise_chains(self, chains, fitnesses):
        fitnesses = deepcopy(fitnesses)
        last_best_chain = chains[0]
        prev_fit = fitnesses[0]
        fig = plt.figure(figsize=(10, 10))
        for ch_id, chain in enumerate(chains):
            self._visualise_chain(chain, title='Current chain')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.chains_imgs.append(img)
            plt.clf()
            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_chain = chain
            prev_fit = fitnesses[ch_id]
            plt.clf()
            self._visualise_chain(last_best_chain, title=f'Best chain after {round(ch_id)} evals')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.best_chains_imgs.append(img)
            plt.clf()
        plt.close('all')

    def _visualise_convergence(self, fitness_history):
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
        fig = plt.figure(figsize=(10, 10))
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        for ts in ts_set:
            ind += 1
            plt.plot(df['ts'], df['fitness'], label='Composer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)
            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.convergence_imgs.append(img)
            plt.clf()
        plt.close('all')

    def visualise_history(self, chains, fitnesses):
        print('START VISUALISATION')
        try:
            self._clean(with_gif=True)
            self._visualise_chains(chains, fitnesses)
            self._visualise_convergence(fitnesses)
            self._merge_images()
            self._combine_gifs()
            self._clean()
        except Exception as ex:
            print(f'Visualisation failed with {ex}')

    def _merge_images(self):
        for i in range(1, len(self.chains_imgs)):
            im1 = self.chains_imgs[i]
            im2 = self.best_chains_imgs[i]
            im3 = self.convergence_imgs[i]
            imgs = (im1, im2, im3)
            merged = np.concatenate(imgs, axis=1)
            self.merged_imgs.append(Image.fromarray(np.uint8(merged)))

    def _combine_gifs(self):
        path = f'{self.temp_path}final_{str(time())}.gif'
        imgs = self.merged_imgs[1:]
        self.merged_imgs[0].save(path, save_all=True, append_images=imgs,
                                 optimize=False, duration=0.5, loop=0)

    def _clean(self, with_gif=False):
        try:
            files = glob(f'{self.temp_path}*.png')
            if with_gif:
                files += glob(f'{self.temp_path}*.gif')
            for file in files:
                remove(file)
        except Exception as ex:
            print(ex)

    def _save_chain_params(self, chain, path):
        cols, data = self._collect_params(chain)
        df = self._create_dataframe(chain, cols, data)
        df.to_csv(path)

    def _collect_params(self, chain):
        columns = ['Model']
        data = {columns[0]: []}
        for i in chain.nodes:
            for param in i.custom_params:
                if i.custom_params != 'default_params':
                    if param not in columns:
                        columns.append(param)
                        data[param] = []
        return columns, data

    def _create_dataframe(self, chain, columns, data):
        for i in chain.nodes:
            data[columns[0]].append(str(i.model.metadata.id))
            for param in list(data.keys())[1:]:
                if i.custom_params == 'default_params':
                    data[param].append('-')
                elif param in i.custom_params:
                    res = i.custom_params[param]
                    if type(res) is float:
                        res = float('{:.5f}'.format(res))
                    if type(res) is np.float64:
                        res = np.round(res, 5)
                    data[param].append(res)
                else:
                    data[param].append('-')
        df = pd.DataFrame(data, columns=columns)
        df = df.set_index('Model')
        df = df.T
        return df


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def colors_by_node_labels(node_labels: dict):
    colors = [color for color in range(len(node_labels.keys()))]
    return colors


def scaled_node_size(nodes_amount):
    size = int(7000.0 / ceil(log2(nodes_amount)))
    return size


def hierarchy_pos(graph, root, levels=None, width=1., height=1.):
    """If there is a cycle that is reachable from root, then this will see infinite recursion.
       graph: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing"""
    total = "total"
    cur = "current"

    def make_levels(levels, node=root, current_level=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not current_level in levels:
            levels[current_level] = {total: 0, cur: 0}
        levels[current_level][total] += 1
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, current_level + 1, node)
        return levels

    def make_pos(pos, node=root, current_level=0, parent=None, vert_loc=0):
        dx = 1 / levels[current_level][total]
        left = dx / 2
        pos[node] = ((left + dx * levels[current_level][cur]) * width, vert_loc)
        levels[current_level][cur] += 1
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, current_level + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {total: levels[l], cur: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})
