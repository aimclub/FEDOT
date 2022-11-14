from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
from networkx import gn_graph

from examples.advanced.abstract_graph_evolution.graph_metrics import get_edit_dist_metric
from fedot.core.adapter.nx_adapter import BaseNetworkxAdapter
from fedot.core.visualisation.graph_viz import GraphVisualiser


def plot_graphs(target_graph, graph):
    adapter = BaseNetworkxAdapter()

    def show_graph(g: nx.DiGraph, ax: plt.Axes):
        GraphVisualiser.draw_nx_dag(adapter.adapt(g), ax,
                                    node_size_scale=0.2, font_size_scale=0.25,
                                    edge_curvature_scale=0.5)

    # 2 subplots
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for g, ax in zip((target_graph, graph), axs):
        show_graph(g, ax)

    start = datetime.now()
    print("Computing metric...")

    metric_f = get_edit_dist_metric(target_graph, timeout=None)
    metric = metric_f(graph)

    end = datetime.now() - start
    print(f'metric: {metric}, computed for '
          f'size {len(target_graph.nodes)} in {end.seconds} sec.')
    plt.title(f'GED: {metric}')
    plt.show()


def try_random(n=10):
    g1 = gn_graph(n)
    g2 = gn_graph(n)
    plot_graphs(g1, g2)


if __name__ == "__main__":
    try_random()
