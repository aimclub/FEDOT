import datetime
import os
from pathlib import Path
from textwrap import wrap
from typing import Optional, Union, Callable, Any, Tuple, List, Dict
from uuid import uuid4

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import ArrowStyle
from pyvis.network import Network
from seaborn import color_palette

from fedot.core.log import default_log
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utils import default_fedot_data_dir


class GraphVisualiser:
    def __init__(self):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        self.log = default_log(self)

    def visualise(self, graph: Union['Graph', 'OptGraph'], save_path: Optional[Union[os.PathLike, str]] = None,
                  engine: str = 'matplotlib', nodes_color: Optional[Union[str, Tuple[float, float, float]]] = None,
                  dpi: int = 300, edges_curvature: float = 0.3):
        if graph.nodes:
            raise ValueError('Empty graph can not be visualized.')
        # Define colors
        if not nodes_color:
            if type(graph).__name__ in ('Pipeline', 'OptGraph'):
                nodes_color = self.__get_colors_by_tags
            else:
                nodes_color = self.__get_colors_by_labels
        if engine == 'matplotlib':
            self.draw_with_networkx(graph, save_path, nodes_color, dpi, edges_curvature)
        elif engine == 'pyvis':
            self.draw_with_pyvis(graph, save_path, nodes_color)
        elif engine == 'graphviz':
            self.draw_with_graphviz(graph, save_path, nodes_color, dpi)
        else:
            raise NotImplementedError(f'Unexpected visualization engine: {engine}. '
                                      'Possible values: matplotlib, pyvis, graphviz.')

    @staticmethod
    def __get_colors_by_tags(labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
        from fedot.core.visualisation.opt_viz import get_palette_based_on_default_tags
        from fedot.core.repository.operation_types_repository import get_opt_node_tag

        palette = get_palette_based_on_default_tags()
        return {label: palette[get_opt_node_tag(label)] for label in labels}

    @staticmethod
    def __get_colors_by_labels(labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
        unique_labels = list(set(labels))
        palette = color_palette('tab10', len(unique_labels))
        return {label: palette[unique_labels.index(label)] for label in labels}

    @staticmethod
    def draw_with_graphviz(graph: Union['Graph', 'OptGraph'], save_path: Optional[Union[os.PathLike, str]] = None,
                           nodes_color=__get_colors_by_tags.__func__, dpi=300):
        nx_graph, nodes = graph_structure_as_nx_graph(graph)
        # Define colors
        if callable(nodes_color):
            colors = nodes_color([str(node) for node in nodes.values()])
        else:
            colors = {str(node): nodes_color for node in nodes.values()}
        for n, data in nx_graph.nodes(data=True):
            label = str(nodes[n])
            data['label'] = label.replace('_', ' ')
            data['color'] = to_hex(colors[label])

        gv_graph = nx.nx_agraph.to_agraph(nx_graph)
        kwargs = {'prog': 'dot', 'args': f'-Gnodesep=0.5 -Gdpi={dpi} -Grankdir="LR"'}

        if save_path:
            gv_graph.draw(save_path, **kwargs)
        else:
            save_path = Path(default_fedot_data_dir(), 'graph_plots', str(uuid4()) + '.png')
            save_path.parent.mkdir(exist_ok=True)
            gv_graph.draw(save_path, **kwargs)

            img = plt.imread(str(save_path))
            plt.imshow(img)
            plt.gca().axis('off')
            plt.gcf().set_dpi(dpi)
            plt.tight_layout()
            plt.show()
            remove_old_files_from_dir(save_path.parent)

    @staticmethod
    def draw_with_pyvis(graph: Union['Graph', 'OptGraph'], save_path: Optional[Union[os.PathLike, str]] = None,
                        nodes_color=__get_colors_by_tags.__func__):
        net = Network('500px', '1000px', directed=True)
        nx_graph, nodes = graph_structure_as_nx_graph(graph)
        # Define colors
        if callable(nodes_color):
            colors = nodes_color([str(node) for node in nodes.values()])
        else:
            colors = {str(node): nodes_color for node in nodes.values()}
        for n, data in nx_graph.nodes(data=True):
            operation = nodes[n]
            label = str(operation)
            data['n_id'] = str(n)
            data['label'] = label.replace('_', ' ')
            params = operation.content.get('params')
            if isinstance(params, dict):
                params = str(params)[1:-1]
            data['title'] = params
            data['level'] = operation.distance_to_primary_level
            data['color'] = to_hex(colors[label])
            data['font'] = '20px'
            data['labelHighlightBold'] = True

        for _, data in nx_graph.nodes(data=True):
            net.add_node(**data)
        for u, v in nx_graph.edges:
            net.add_edge(str(u), str(v))

        if save_path:
            net.save_graph(str(save_path))
            return
        save_path = Path(default_fedot_data_dir(), 'graph_plots', str(uuid4()) + '.html')
        save_path.parent.mkdir(exist_ok=True)
        net.show(str(save_path))
        remove_old_files_from_dir(save_path.parent)

    def draw_with_networkx(self, graph: Union['Graph', 'OptGraph'], save_path=None,
                           nodes_color: Optional[Union[str, Tuple[float, float, float],
                                                       Callable[
                                                           [List[str]], Dict[str, Tuple[float, float, float]]]]] = None,
                           dpi: int = 300, edges_curvature: float = 0.3,
                           in_graph_converter_function: Callable = graph_structure_as_nx_graph):
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        self.draw_nx_dag(graph, ax, nodes_color, edges_curvature, in_graph_converter_function)
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path, dpi=dpi)
            plt.close()

    def draw_nx_dag(self, graph: Union['Graph', 'OptGraph'], ax: Optional[plt.Axes] = None,
                    nodes_color: Optional[Union[str, Tuple[float, float, float],
                                                Callable[[List[str]], Dict[str, Tuple[float, float, float]]]]] = None,
                    edges_curvature: float = 0.3,
                    in_graph_converter_function: Callable = graph_structure_as_nx_graph):

        def get_scaled_node_size(nodes_amount):
            min_size = 1000
            max_size = 4000
            size = min_size + int((max_size - min_size) / np.log2(max(nodes_amount, 2)))
            return size

        if ax is None:
            ax = plt.gca()

        nx_graph, nodes = in_graph_converter_function(graph)
        # Define colors
        if callable(nodes_color):
            colors = nodes_color([str(node) for node in nodes.values()])
            edge_colors = [colors[str(node)] for node in nodes.values()]
        else:
            edge_colors = nodes_color
        # Define hierarchy_level
        for u, v, e in nx_graph.edges(data=True):
            for node_id in (u, v):
                nx_graph.nodes[node_id]['hierarchy_level'] = nodes[node_id].distance_to_primary_level
        # Get nodes positions
        pos, longest_sequence = get_hierarchy_pos(nx_graph)
        node_size = get_scaled_node_size(longest_sequence)
        # Draw nodes
        nx.draw_networkx_nodes(nx_graph, pos, node_size=node_size, ax=ax, node_color='w', linewidths=3,
                               edgecolors=edge_colors)
        # Define edges curvature
        connection_style = 'arc3'
        curved_connection_style = connection_style + ',rad={}'
        for u, v, e in nx_graph.edges(data=True):
            e['connectionstyle'] = connection_style
            p1, p2 = np.array(pos[u]), np.array(pos[v])
            min_distance = 1
            closest_node_id = None
            for node_id in nx_graph.nodes:
                if node_id in (u, v):
                    continue
                p3 = np.array(pos[node_id])
                distance = abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
                if distance > 0.15:
                    continue
                min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
                min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
                if not (min_x <= p3[0] <= max_x and min_y <= p3[1] <= max_y):
                    continue
                if distance > min_distance:
                    continue
                min_distance = distance
                closest_node_id = node_id

            if closest_node_id is None:
                continue
            p3 = pos[closest_node_id]
            curvature_factor = (1 / (min_distance + 1)) ** 2
            if p1[1] == p2[1]:
                curvature_sign = -1
            else:
                k = np.divide(*(p1 - p2))
                b = p1[1] - k * p1[0]
                sign_pow = p3[1] > k * p3[0] + b
                curvature_sign = (-1) ** sign_pow
            e['connectionstyle'] = curved_connection_style.format(edges_curvature * curvature_factor * curvature_sign)
        # Draw edges
        arrow_style = ArrowStyle('Simple', head_length=1.5, head_width=0.8)
        for u, v, e in nx_graph.edges(data=True):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], node_size=node_size, ax=ax, arrowsize=10,
                                   arrowstyle=arrow_style, connectionstyle=e['connectionstyle'])
        # Rescale graph for all nodes to fit in
        x_1, x_2 = ax.get_xlim()
        y_1, y_2 = ax.get_ylim()
        offset = 0.2
        x_offset = x_2 * offset
        y_offset = y_2 * offset
        ax.set_xlim(x_1 - x_offset, x_2 + x_offset)
        ax.set_ylim(y_1 - y_offset, y_2 + y_offset)
        ax.axis('off')

        self._draw_nx_labels(pos, {node_id: str(node) for node_id, node in nodes.items()}, ax, longest_sequence)

        plt.tight_layout()

    @staticmethod
    def _draw_nx_labels(pos, node_labels, ax, max_sequence_length):
        def get_scaled_font_size(nodes_amount):
            min_size = 6
            max_size = 16

            size = min_size + int((max_size - min_size) / np.log2(max(nodes_amount, 2)))
            return size

        if ax is None:
            ax = plt.gca()
        for node, (x, y) in pos.items():
            text = '\n'.join(wrap(node_labels[node].replace('_', ' ').replace('-', ' '), 10))
            ax.text(x, y, text, ha='center', va='center', fontsize=get_scaled_font_size(max_sequence_length),
                    bbox=dict(alpha=0.9, color='w', boxstyle='round'))


def get_hierarchy_pos(graph: nx.DiGraph, max_line_length: int = 6) -> Tuple[Dict[Any, Tuple[float, float]], int]:
    """By default, returns 'networkx.multipartite_layout' positions based on 'hierarchy_level` from node data - the
     property must be set beforehand.
    If line of nodes reaches 'max_line_length', the result is the combination of 'networkx.shell_layout' and
    'networkx.spring_layout'.
    :param graph: the graph.
    :param max_line_length: the limit for common nodes horizontal or vertical line.
    """
    longest_path = nx.dag_longest_path(graph, weight=None)
    longest_sequence = len(longest_path)

    if longest_sequence > max_line_length:
        layers = {}
        for n, data in graph.nodes(data=True):
            distance = data['hierarchy_level']
            layers[distance] = layers.get(distance, []) + [n]
        nlist = [layers[layer_num] for layer_num in sorted(layers.keys())]
        pos = nx.shell_layout(graph, nlist=nlist)

        pos = {n: np.array(x_y) + (np.random.random(2) - 0.5) * 0.01 for n, x_y in pos.items()}
        pos = nx.spring_layout(graph, k=100, pos=pos, center=(0.5, 0.5), seed=42, scale=-1)

    else:
        pos = nx.multipartite_layout(graph, subset_key='hierarchy_level')

    return pos, longest_sequence


def remove_old_files_from_dir(dir: Path, time_interval=datetime.timedelta(minutes=10)):
    for path in dir.iterdir():
        if datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_ctime) > time_interval:
            path.unlink()
