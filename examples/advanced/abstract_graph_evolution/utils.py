from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
from networkx import gn_graph

from examples.advanced.abstract_graph_evolution.graph_metrics import (
    get_edit_dist_metric,
    matrix_edit_dist,
    spectral_dist
)
from fedot.core.adapter.nx_adapter import BaseNetworkxAdapter
from fedot.core.optimisers.objective import Objective
from fedot.core.visualisation.graph_viz import GraphVisualiser


def measure_graphs(target_graph, graph, vis=False):
    adapter = BaseNetworkxAdapter()

    ged = get_edit_dist_metric(target_graph, timeout=None)
    objective = Objective(quality_metrics={
        'edit_distance': ged,
        'matrix_edit_dist': partial(matrix_edit_dist, target_graph),
        'spectral_laplacian': partial(spectral_dist, target_graph),
    })

    start = datetime.now()
    print("Computing metric...")
    fitness = objective(graph)
    end = datetime.now() - start
    print(f'metrics: {fitness}, computed for '
          f'size {len(target_graph.nodes)} in {end.seconds} sec.')

    if vis:
        def show_graph(g: nx.DiGraph, ax: plt.Axes):
            GraphVisualiser.draw_nx_dag(adapter.adapt(g), ax,
                                        node_size_scale=0.2, font_size_scale=0.25,
                                        edge_curvature_scale=0.5)
        # 2 subplots
        fig, axs = plt.subplots(nrows=1, ncols=2)
        for g, ax in zip((target_graph, graph), axs):
            show_graph(g, ax)

        plt.title(f'metrics: {fitness.values}')
        plt.show()


def try_random(n=20, it=1):
    for i in range(it):
        g1 = gn_graph(n)
        g2 = gn_graph(n)
        measure_graphs(g1, g2, vis=False)


if __name__ == "__main__":
    try_random(it=10)
