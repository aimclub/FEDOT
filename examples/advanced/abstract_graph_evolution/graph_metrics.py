from datetime import timedelta
from typing import Optional, Callable, Dict

import netcomp
import networkx as nx
import numpy as np
from netcomp import laplacian_matrix, normalized_laplacian_eig
from netcomp.linalg import _eigs
from networkx import graph_edit_distance

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         requirements: Optional[PipelineComposerRequirements] = None,
                         timeout=timedelta(seconds=60),
                         ) -> Callable[[nx.DiGraph], float]:

    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_pipeline_fit_time
    else:
        upper_bound = None

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.seconds if timeout else None,
                                 )
        return ged or upper_bound

    return metric


def size_diff(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    nodes_diff = abs(target_graph.number_of_nodes() - graph.number_of_nodes())
    edges_diff = abs(target_graph.number_of_edges() - graph.number_of_edges())
    # return nodes_diff + edges_diff
    return np.sqrt(nodes_diff) + np.sqrt(edges_diff)


def matrix_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)
    nmin, nmax = min_max(target_adj.shape[0], adj.shape[0])
    if nmin != nmax:
        shape = (nmax, nmax)
        target_adj.resize(shape)
        adj.resize(shape)
    value = netcomp.edit_distance(target_adj, adj)
    return value


def spectral_dist(target_graph: nx.DiGraph, graph: nx.DiGraph,
                  k: int = 20, kind: str = 'laplacian',
                  with_num_nodes_penalty: bool = False,
                  match_size: bool = False,
                  ) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)

    # compute spectral distance
    value = lambda_dist(target_adj, adj, kind=kind, k=k, match_size=match_size)

    if with_num_nodes_penalty:
        value += size_diff(target_graph, graph)
    return value


def spectral_dists_all(target_graph: nx.DiGraph, graph: nx.DiGraph,
                       k: int = 20, match_size: bool = True) -> dict:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)

    print(f'computing metrics for {k} spectral values between {target_adj.shape} & {adj.shape}')

    vals = {}
    for kind in ('adjacency', 'laplacian_norm', 'laplacian'):
        value = lambda_dist(target_adj, adj, kind=kind, k=k, match_size=match_size)
        vals[kind] = np.round(value, 3)
    vals['nodes_diff'] = size_diff(target_graph, graph)
    return vals


def min_max(a, b):
    return (a, b) if a <= b else (b, a)


def lambda_dist(A1, A2, k=None, p=2, kind='laplacian', match_size=True):
    """The lambda distance between graphs, which is defined as

        d(G1,G2) = norm(L_1 - L_2)

    where L_1 is a vector of the top k eigenvalues of the appropriate matrix
    associated with G1, and L2 is defined similarly.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    k : Integer
        The number of eigenvalues to be compared

    p : non-zero Float
        The p-norm is used to compare the resulting vector of eigenvalues.

    kind : String , in {'laplacian','laplacian_norm','adjacency'}
        The matrix for which eigenvalues will be calculated.

    Returns
    -------
    dist : float
        The distance between the two graphs

    Notes
    -----
    The norm can be any p-norm; by default we use p=2. If p<0 is used, the
    result is not a mathematical norm, but may still be interesting and/or
    useful.

    If k is provided, then we use the k SMALLEST eigenvalues for the Laplacian
    distances, and we use the k LARGEST eigenvalues for the adjacency
    distance. This is because the corresponding order flips, as L = D-A.

    References
    ----------

    See Also
    --------
    netcomp.linalg._eigs
    normalized_laplacian_eigs

    """
    # check sizes & determine number of eigenvalues (k)
    nmin, nmax = min_max(A1.shape[0], A2.shape[0])
    if match_size:
        shape = (nmax, nmax)
        A1.resize(shape)
        A2.resize(shape)
    else:
        k = min(k, nmin)

    if kind == 'laplacian':
        # form matrices
        L1, L2 = [laplacian_matrix(A) for A in [A1, A2]]
        # get eigenvalues, ignore eigenvectors
        evals1, evals2 = [_eigs(L)[0] for L in [L1, L2]]
    elif kind == 'laplacian_norm':
        # use our function to graph evals of normalized laplacian
        evals1, evals2 = [normalized_laplacian_eig(A)[0] for A in [A1, A2]]
    elif kind == 'adjacency':
        evals1, evals2 = [_eigs(A)[0] for A in [A1, A2]]
        # reverse, so that we are sorted from large to small, since we care
        # about the k LARGEST eigenvalues for the adjacency distance
        evals1, evals2 = [evals[::-1] for evals in [evals1, evals2]]
    else:
        raise AttributeError(f"Invalid type {kind}, choose from 'laplacian', "
                             f"'laplacian_norm', and 'adjacency'.")
    dist = np.linalg.norm(evals1[:k] - evals2[:k], ord=p)
    return dist
