import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from functools import partial
from itertools import product
from typing import Callable, Sequence, Optional, Dict

import networkx as nx
from networkx import graph_edit_distance, gn_graph

from examples.advanced.abstract_graph_evolution.abstract_graph_search import get_similarity_metric
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

    print("Computing GED...")
    metric_f = get_similarity_metric(target_graph)
    metric = metric_f(graph)
    plt.title(f'GED: {metric}')

    plt.show()


def try_random(n=50):
    g1 = gn_graph(n)
    g2 = gn_graph(n)
    plot_graphs(g1, g2)


if __name__ == "__main__":
    try_random()
