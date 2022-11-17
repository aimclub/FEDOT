from __future__ import annotations

import datetime
import os
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
from uuid import uuid4

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import ArrowStyle
from pyvis.network import Network
from seaborn import color_palette

from fedot.core.dag.graph_utils import distance_to_primary_level
from fedot.core.dag.convert import graph_structure_as_nx_graph
from fedot.core.log import default_log
from fedot.core.utils import default_fedot_data_dir

if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph
    from fedot.core.optimisers.graph import OptGraph

    GraphType = Union[Graph, OptGraph]

PathType = Union[os.PathLike, str]

MatplotlibColorType = Union[str, Sequence[float]]
LabelsColorMapType = Dict[str, MatplotlibColorType]
NodeColorFunctionType = Callable[[Iterable[str]], LabelsColorMapType]
NodeColorType = Union[MatplotlibColorType, LabelsColorMapType, NodeColorFunctionType]


class GraphVisualizer:
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None):
        visuals_params = visuals_params or {}
        default_visuals_params = dict(
            engine='matplotlib',
            dpi=100,
            node_color=self.__get_colors_by_labels,
            node_size_scale=1.0,
            font_size_scale=1.0,
            edge_curvature_scale=1.0,
            graph_to_nx_convert_func=graph_structure_as_nx_graph
        )
        default_visuals_params.update(visuals_params)
        self.visuals_params = default_visuals_params
        self.graph = graph

        self.log = default_log(self)

    def visualise(self, save_path: Optional[PathType] = None, engine: Optional[str] = None,
                  node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                  node_size_scale: Optional[float] = None,
                  font_size_scale: Optional[float] = None, edge_curvature_scale: Optional[float] = None):
        engine = engine or self.get_predefined_value('engine')

        if not self.graph.nodes:
            raise ValueError('Empty graph can not be visualized.')

        if engine == 'matplotlib':
            self.__draw_with_networkx(save_path, node_color, dpi, node_size_scale, font_size_scale,
                                      edge_curvature_scale)
        elif engine == 'pyvis':
            self.__draw_with_pyvis(save_path, node_color)
        elif engine == 'graphviz':
            self.__draw_with_graphviz(save_path, node_color, dpi)
        else:
            raise NotImplementedError(f'Unexpected visualization engine: {engine}. '
                                      'Possible values: matplotlib, pyvis, graphviz.')

    @staticmethod
    def __get_colors_by_labels(labels: Iterable[str]) -> LabelsColorMapType:
        unique_labels = list(set(labels))
        palette = color_palette('tab10', len(unique_labels))
        return {label: palette[unique_labels.index(label)] for label in labels}

    def __draw_with_graphviz(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
                             dpi: Optional[int] = None, graph_to_nx_convert_func: Optional[Callable] = None):
        save_path = save_path or self.get_predefined_value('save_path')
        node_color = node_color or self.get_predefined_value('node_color')
        dpi = dpi or self.get_predefined_value('dpi')
        graph_to_nx_convert_func = graph_to_nx_convert_func or self.get_predefined_value('graph_to_nx_convert_func')

        nx_graph, nodes = graph_to_nx_convert_func(self.graph)
        # Define colors
        if callable(node_color):
            colors = node_color([str(node) for node in nodes.values()])
        elif isinstance(node_color, dict):
            colors = node_color
        else:
            colors = {str(node): node_color for node in nodes.values()}
        for n, data in nx_graph.nodes(data=True):
            label = str(nodes[n])
            data['label'] = label.replace('_', ' ')
            data['color'] = to_hex(colors.get(label, colors.get(None)))

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

    def __draw_with_pyvis(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
                          graph_to_nx_convert_func: Optional[Callable] = None):
        save_path = save_path or self.get_predefined_value('save_path')
        node_color = node_color or self.get_predefined_value('node_color')
        graph_to_nx_convert_func = graph_to_nx_convert_func or self.get_predefined_value('graph_to_nx_convert_func')

        net = Network('500px', '1000px', directed=True)
        nx_graph, nodes = graph_to_nx_convert_func(self.graph)
        # Define colors
        if callable(node_color):
            colors = node_color([str(node) for node in nodes.values()])
        elif isinstance(node_color, dict):
            colors = node_color
        else:
            colors = {str(node): node_color for node in nodes.values()}
        for n, data in nx_graph.nodes(data=True):
            operation = nodes[n]
            label = str(operation)
            data['n_id'] = str(n)
            data['label'] = label.replace('_', ' ')
            params = operation.content.get('params')
            if isinstance(params, dict):
                params = str(params)[1:-1]
            data['title'] = params
            data['level'] = distance_to_primary_level(operation)
            data['color'] = to_hex(colors.get(label, colors.get(None)))
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

    def __draw_with_networkx(self, save_path: Optional[PathType] = None,
                             node_color: Optional[NodeColorType] = None,
                             dpi: Optional[int] = None, node_size_scale: Optional[float] = None,
                             font_size_scale: Optional[float] = None, edge_curvature_scale: Optional[float] = None,
                             graph_to_nx_convert_func: Optional[Callable] = None):
        save_path = save_path or self.get_predefined_value('save_path')
        node_color = node_color or self.get_predefined_value('node_color')
        dpi = dpi or self.get_predefined_value('dpi')
        node_size_scale = node_size_scale or self.get_predefined_value('node_size_scale')
        font_size_scale = font_size_scale or self.get_predefined_value('font_size_scale')
        edge_curvature_scale = (edge_curvature_scale if edge_curvature_scale is not None
                                else self.get_predefined_value('edge_curvature_scale'))
        graph_to_nx_convert_func = graph_to_nx_convert_func or self.get_predefined_value('graph_to_nx_convert_func')

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_dpi(dpi)
        self.draw_nx_dag(self.graph, ax, node_color, node_size_scale, font_size_scale, edge_curvature_scale,
                         graph_to_nx_convert_func)
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path, dpi=dpi)
            plt.close()

    @staticmethod
    def draw_nx_dag(graph: GraphType, ax: Optional[plt.Axes] = None,
                    node_color: Optional[NodeColorType] = None,
                    node_size_scale: float = 1, font_size_scale: float = 1, edge_curvature_scale: float = 1,
                    graph_to_nx_convert_func: Callable = graph_structure_as_nx_graph):

        def draw_nx_labels(pos, node_labels, ax, max_sequence_length, font_size_scale=1.0):
            def get_scaled_font_size(nodes_amount):
                min_size = 2
                max_size = 20

                size = min_size + int((max_size - min_size) / np.log2(max(nodes_amount, 2)))
                return size

            if ax is None:
                ax = plt.gca()
            for node, (x, y) in pos.items():
                text = '\n'.join(wrap(node_labels[node].replace('_', ' ').replace('-', ' '), 10))
                ax.text(x, y, text, ha='center', va='center',
                        fontsize=get_scaled_font_size(max_sequence_length) * font_size_scale,
                        bbox=dict(alpha=0.9, color='w', boxstyle='round'))

        def get_scaled_node_size(nodes_amount):
            min_size = 500
            max_size = 5000
            size = min_size + int((max_size - min_size) / np.log2(max(nodes_amount, 2)))
            return size

        if ax is None:
            ax = plt.gca()

        nx_graph, nodes = graph_to_nx_convert_func(graph)
        # Define colors
        if callable(node_color):
            node_color = node_color([str(node) for node in nodes.values()])
        if isinstance(node_color, dict):
            node_color = [node_color.get(str(node), node_color.get(None)) for node in nodes.values()]
        # Define hierarchy_level
        for node_id, node_data in nx_graph.nodes(data=True):
            node_data['hierarchy_level'] = distance_to_primary_level(nodes[node_id])
        # Get nodes positions
        pos, longest_sequence = get_hierarchy_pos(nx_graph)
        node_size = get_scaled_node_size(longest_sequence) * node_size_scale
        # Draw the graph's nodes.
        nx.draw_networkx_nodes(nx_graph, pos, node_size=node_size, ax=ax, node_color='w', linewidths=3,
                               edgecolors=node_color)
        # Draw the graph's node labels.
        draw_nx_labels(pos, {node_id: str(node) for node_id, node in nodes.items()}, ax, longest_sequence,
                       font_size_scale)
        # The ongoing section defines curvature for all edges.
        #   This is 'connection style' for an edge that does not intersect any nodes.
        connection_style = 'arc3'
        #   This is 'connection style' template for an edge that is too close to any node and must bend around it.
        #   The curvature value is defined individually for each edge.
        connection_style_curved_template = connection_style + ',rad={}'
        default_edge_curvature = 0.3
        #   The minimum distance from a node to an edge on which the edge must bend around the node.
        node_distance_gap = 0.15
        for u, v, e in nx_graph.edges(data=True):
            e['connectionstyle'] = connection_style
            p_1, p_2 = np.array(pos[u]), np.array(pos[v])
            p_1_2 = p_2 - p_1
            p_1_2_length = np.linalg.norm(p_1_2)
            # Finding the closest node to the edge.
            min_distance_found = node_distance_gap * 2  # It just must be bigger than the gap.
            closest_node_id = None
            for node_id in nx_graph.nodes:
                if node_id in (u, v):
                    continue  # The node is adjacent to the edge.
                p_3 = np.array(pos[node_id])
                distance_to_node = abs(np.cross(p_1_2, p_3 - p_1)) / p_1_2_length
                if (distance_to_node > min(node_distance_gap, min_distance_found)  # The node is too far.
                        or ((p_3 - p_1) @ p_1_2) < 0  # There's no perpendicular from the node to the edge.
                        or ((p_3 - p_2) @ -p_1_2) < 0):
                    continue
                min_distance_found = distance_to_node
                closest_node_id = node_id

            if closest_node_id is None:
                continue  # There's no node to bend around.
            # Finally, define the edge's curvature based on the closest node position.
            p_3 = np.array(pos[closest_node_id])
            p_1_3 = p_3 - p_1
            curvature_strength = default_edge_curvature * edge_curvature_scale
            # 'alpha' denotes the angle between the abscissa and the edge.
            cos_alpha = p_1_2[0] / p_1_2_length
            sin_alpha = np.sqrt(1 - cos_alpha ** 2) * (-1) ** (p_1_2[1] < 0)
            # The closest node is placed as if the edge matched the abscissa.
            # Then, its ordinate shows on which side of the edge it is, "on the left" or "on the right".
            rotation_matrix = np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]])
            p_1_3_rotated = rotation_matrix @ p_1_3
            curvature_direction = (-1) ** (p_1_3_rotated[1] < 0)  # +1 is a "boat" \/, -1 is a "cat" /\.
            edge_curvature = curvature_direction * curvature_strength
            e['connectionstyle'] = connection_style_curved_template.format(edge_curvature)
        # Draw the graph's edges.
        arrow_style = ArrowStyle('Simple', head_length=1.5, head_width=0.8)
        for u, v, e in nx_graph.edges(data=True):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], node_size=node_size, ax=ax, arrowsize=10,
                                   arrowstyle=arrow_style, connectionstyle=e['connectionstyle'])
        # Rescale the figure for all nodes to fit in.
        x_1, x_2 = ax.get_xlim()
        y_1, y_2 = ax.get_ylim()
        offset = 0.2
        x_offset = x_2 * offset
        y_offset = y_2 * offset
        ax.set_xlim(x_1 - x_offset, x_2 + x_offset)
        ax.set_ylim(y_1 - y_offset, y_2 + y_offset)
        ax.axis('off')
        plt.tight_layout()

    def get_predefined_value(self, param: str):
        return self.visuals_params.get(param)


def get_hierarchy_pos(graph: nx.DiGraph, max_line_length: int = 6) -> Tuple[Dict[Any, Tuple[float, float]], int]:
    """By default, returns 'networkx.multipartite_layout' positions based on 'hierarchy_level` from node data - the
     property must be set beforehand.
    If line of nodes reaches 'max_line_length', the result is the combination of 'networkx.multipartite_layout' and
    'networkx.spring_layout'.
    :param graph: the graph.
    :param max_line_length: the limit for common nodes horizontal or vertical line.
    """
    longest_path = nx.dag_longest_path(graph, weight=None)
    longest_sequence = len(longest_path)

    pos = nx.multipartite_layout(graph, subset_key='hierarchy_level')

    y_level_nodes_count = {}
    for x, _ in pos.values():
        y_level_nodes_count[x] = y_level_nodes_count.get(x, 0) + 1
        nodes_on_level = y_level_nodes_count[x]
        if nodes_on_level > longest_sequence:
            longest_sequence = nodes_on_level

    if longest_sequence > max_line_length:
        pos = {n: np.array(x_y) + (np.random.random(2) - 0.5) * 0.001 for n, x_y in pos.items()}
        pos = nx.spring_layout(graph, k=2, iterations=5, pos=pos, seed=42)

    return pos, longest_sequence


def remove_old_files_from_dir(dir_: Path, time_interval=datetime.timedelta(minutes=10)):
    for path in dir_.iterdir():
        if datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_ctime) > time_interval:
            path.unlink()
