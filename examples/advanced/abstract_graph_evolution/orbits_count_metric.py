### Implementation adopted from https://github.com/snap-stanford/GraphRNN/
import os
import subprocess
import tempfile
from tempfile import TemporaryFile

import networkx as nx
import numpy as np

from examples.advanced.abstract_graph_evolution import mmd

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
        '3path' : [1, 2],
        '4cycle' : [8],
}
COUNT_START_STR = 'orbit counts: \n'


def _edge_list_reindexed(graph: nx.Graph):
    idx = 0
    id2idx = dict()
    for u in graph.nodes():
        id2idx[str(u)] = idx
        idx += 1

    for (u, v) in graph.edges():
        yield id2idx[str(u)], id2idx[str(v)]


def orca_run(graph: nx.Graph,
             orbit_n: int = 4,
             orca_bin_path: os.PathLike = './eval/orca/orca'):
    tmp_fname = tempfile.mktemp(prefix='orca_compute.txt')
    with open(tmp_fname, 'w') as f:
        f.write(f'{str(graph.number_of_nodes())} {str(graph.number_of_edges())}\n')
        for (u, v) in _edge_list_reindexed(graph):
            f.write(f'{u} {v}\n')

    output = subprocess.check_output([orca_bin_path, 'node', str(orbit_n), tmp_fname, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats_graph(graph, motif_type='4cycle', all_motifs=False):
    # Return graph motif counts (int for each graph) normalized by graph size

    orbit_counts = orca_run(graph)

    if all_motifs:
        motif_counts = np.sum(orbit_counts, axis=0)
    else:
        indices = motif_to_indices[motif_type]
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)
        motif_counts = np.sum(motif_counts)
    motif_normalized = motif_counts / graph.number_of_nodes()

    return motif_normalized
