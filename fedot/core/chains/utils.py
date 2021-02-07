from uuid import uuid4

import networkx as nx


def chain_as_nx_graph(chain: 'Chain'):
    graph = nx.DiGraph()
    node_labels = {}
    new_node_idx = {}
    for node in chain.nodes:
        unique_id, label = uuid4(), node
        node_labels[unique_id] = node
        new_node_idx[node] = unique_id
        graph.add_node(unique_id)

    def add_edges(graph, chain, new_node_idx):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(new_node_idx[child], new_node_idx[node])

    add_edges(graph, chain, new_node_idx)
    return graph, node_labels
