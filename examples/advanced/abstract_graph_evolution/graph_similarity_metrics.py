from collections.abc import Sequence

import numpy as np
from typing import Callable, Tuple

from examples.advanced.abstract_graph_evolution import mmd
from examples.advanced.abstract_graph_evolution.mmd import compute_mmd
from examples.advanced.abstract_graph_evolution.orbits_count_metric import motif_stats_graph
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.objective import ObjectiveFunction

import networkx as nx

from fedot.core.pipelines.convert import graph_structure_as_nx_graph


def compute_all_mmd_stats(graph_prediction: Sequence[Graph],
                          graph_target: Sequence[Graph]) -> Tuple[float, float, float]:

    def transform(graph: Graph) -> nx.Graph:
        return graph_structure_as_nx_graph(graph)[0]

    nx_graph_prediction = list(map(transform, graph_prediction))
    nx_graph_target = list(map(transform, graph_target))

    mmd_degree = degree_stats_impl(nx_graph_prediction, nx_graph_target)
    mmd_clustering = clustering_stats_impl(nx_graph_prediction, nx_graph_target)
    mmd_motifs = motif_stats(nx_graph_prediction, nx_graph_target)

    return mmd_degree, mmd_clustering, mmd_motifs


def degree_stats_impl(graph_prediction: Sequence[nx.Graph],
                      graph_target: Sequence[nx.Graph]) -> float:
    return mmd_stats_impl(nx.degree_histogram, graph_prediction, graph_target, normalize=True)


def clustering_stats_impl(graph_prediction: Sequence[nx.Graph],
                          graph_target: Sequence[nx.Graph]) -> float:
    bins = 100
    return mmd_stats_impl(clustering_stats_graph, graph_prediction, graph_target,
                          sigma=0.1, distance_scaling=bins, normalize=False)


def motif_stats(graph_prediction: Sequence[nx.Graph],
                graph_target: Sequence[nx.Graph]) -> float:
    graph_prediction = [G for G in graph_prediction if G.number_of_nodes() > 0]

    return mmd_stats_impl(motif_stats_graph, graph_prediction, graph_target,
                          kernel=mmd.gaussian, normalize=False)


def mmd_stats_impl(stat_function: Callable[[nx.Graph], np.ndarray],
                   graph_prediction: Sequence[nx.Graph],
                   graph_target: Sequence[nx.Graph],
                   kernel: Callable = mmd.gaussian_emd,
                   sigma: float = 1.0,
                   distance_scaling: float = 1.0,
                   normalize: bool = False) -> float:

    sample_predict = list(map(stat_function, graph_prediction))
    sample_target = list(map(stat_function, graph_target))

    mmd_dist = compute_mmd(sample_target, sample_predict,
                           normalize=normalize, kernel=kernel,
                           sigma=sigma, distance_scaling=distance_scaling)
    return mmd_dist


def clustering_stats_graph(graph: nx.Graph, bins: int = 100) -> np.ndarray:
    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=True)
    return hist
