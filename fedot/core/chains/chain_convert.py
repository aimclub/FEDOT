from uuid import uuid4

import networkx as nx

from fedot.core.chains.chain_template import ChainTemplate


def chain_as_nx_graph(chain: 'Chain'):
    """ Convert FEDOT chain into networkx graph object """
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


def chain_template_as_nx_graph(chain: ChainTemplate):
    """ Convert FEDOT chain template into networkx graph object """
    graph = nx.DiGraph()
    node_labels = {}
    for operation in chain.operation_templates:
        unique_id, label = operation.operation_id, operation.operation_type
        node_labels[unique_id] = label
        graph.add_node(unique_id)

    def add_edges(graph, chain):
        for operation in chain.operation_templates:
            if operation.nodes_from is not None:
                for child in operation.nodes_from:
                    graph.add_edge(child, operation.operation_id)

    add_edges(graph, chain)
    return graph, node_labels
