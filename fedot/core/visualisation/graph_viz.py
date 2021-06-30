import os
from math import ceil, log2
from typing import Optional

import networkx as nx
from matplotlib import pyplot as plt

from fedot.core.log import Log, default_log
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utils import default_fedot_data_dir


class GraphVisualiser:
    def __init__(self, log: Log = default_log(__name__)):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        self.log = log

    def visualise(self, pipeline: 'Graph', save_path: Optional[str] = None):
        try:
            fig, axs = plt.subplots(figsize=(9, 9))
            fig.suptitle('Current graph')
            self.draw_single_graph(pipeline, axs)
            if not save_path:
                plt.show()
            else:
                plt.savefig(save_path)
                plt.close()
        except Exception as ex:
            self.log.error(f'Visualisation failed with {ex}')

    def draw_single_graph(self, graph: 'Graph', ax=None, title=None,
                          in_graph_converter_function=graph_structure_as_nx_graph):
        if type(graph).__name__ == 'Pipeline':
            pos, node_labels = self._draw_tree(graph, ax, title, in_graph_converter_function)
        else:
            pos, node_labels = self._draw_dag(graph, ax, title, in_graph_converter_function)

        self._draw_labels(pos, node_labels, ax)

    def _draw_tree(self, graph: 'Graph', ax=None, title=None,
                   in_graph_converter_function=graph_structure_as_nx_graph):
        nx_graph, node_labels = in_graph_converter_function(graph)
        word_labels = [str(node) for node in node_labels.values()]
        inv_map = {v: k for k, v in node_labels.items()}
        if type(graph).__name__ == 'Pipeline':
            root = inv_map[graph.root_node]
        else:
            root = 0
        minimum_spanning_tree = nx.minimum_spanning_tree(nx_graph.to_undirected())
        pos = hierarchy_pos(minimum_spanning_tree, root=root)
        min_size = 3000
        node_sizes = [min_size for _ in word_labels]
        if title:
            plt.title(title)
        colors = colors_by_node_labels(node_labels)
        nx.draw(nx_graph, pos=pos, with_labels=False,
                node_size=node_sizes, width=2.0,
                node_color=colors, cmap='Set3', ax=ax)
        return pos, node_labels

    def _draw_dag(self, graph: 'Graph', ax=None, title=None,
                  in_graph_converter_function=graph_structure_as_nx_graph):
        nx_graph, node_labels = in_graph_converter_function(graph)
        word_labels = [str(node) for node in node_labels.values()]

        pos = nx.circular_layout(nx_graph)

        min_size = 3000
        node_sizes = [min_size for _ in word_labels]
        if title:
            plt.title(title)
        colors = colors_by_node_labels(node_labels)
        nx.draw(nx_graph, pos=pos, with_labels=False,
                node_size=node_sizes, width=2.0,
                node_color=colors, cmap='Set3', ax=ax)
        return pos, node_labels

    def _draw_labels(self, pos, node_labels, ax):
        for node, (x, y) in pos.items():
            text = '\n'.join(str(node_labels[node]).split('_'))
            if ax is None:
                ax = plt
            ax.text(x, y, text, ha='center', va='center')


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
        if current_level not in levels:
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
        levels = {level: {total: levels[level], cur: 0} for level in levels}
    vert_gap = height / (max([level for level in levels]) + 1)
    return make_pos({})
