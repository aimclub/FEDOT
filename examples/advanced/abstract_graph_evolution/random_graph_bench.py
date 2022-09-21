from collections.abc import Sequence

import numpy as np
from typing import Callable

from examples.advanced.abstract_graph_evolution.mmd import compute_mmd
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.objective import ObjectiveFunction

import networkx as nx


def generate_random_target():
    pass


def degree_stats(graph_prediction: Sequence[Graph], graph_target: Sequence[Graph]) -> ObjectiveFunction:
    pass


def degree_stats_impl(graph_prediction: Sequence[nx.Graph],
                      graph_target: Sequence[nx.Graph]):
    mmd_dist = mmd_stats_impl(nx.degree_histogram, graph_prediction, graph_target, normalize=True)
    return mmd_dist


def clustering_stats_impl(graph_prediction: Sequence[nx.Graph],
                          graph_target: Sequence[nx.Graph]):
    bins = 100
    mmd_dist = mmd_stats_impl(clustering_stats_graph, graph_prediction, graph_target,
                              sigma=0.1, distance_scaling=bins)
    return mmd_dist


def mmd_stats_impl(stat_function: Callable[[nx.Graph], np.ndarray],
                   graph_prediction: Sequence[nx.Graph],
                   graph_target: Sequence[nx.Graph],
                   sigma: float = 1.0,
                   distance_scaling: float = 1.0,
                   normalize: bool = False):
    sample_predict = list(map(stat_function, graph_prediction))
    sample_target = list(map(stat_function, graph_target))
    mmd_dist = compute_mmd(sample_target, sample_predict,
                           normalize=normalize,
                           sigma=sigma, distance_scaling=distance_scaling)
    return mmd_dist


def clustering_stats_graph(graph: nx.Graph, bins: int = 100) -> np.ndarray:
    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=True)
    return hist