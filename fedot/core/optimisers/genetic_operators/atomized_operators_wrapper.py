from typing import Dict

from golem.core.optimisers.graph import OptGraph


def extract_graphs_from_atomized(graph: OptGraph) -> Dict[str, OptGraph]:
    """ Get all graphs from graph with atomized nodes
        Return dict with key as node uid (where graph is stored in atomized models)
        and values as graphs """
    graphs = {'': graph}
    for node in graph.nodes:
        if 'pipeline' in node.parameters:
            extracted_graphs = extract_graphs_from_atomized(node.parameters['pipeline'])
            for k, v in extracted_graphs.items():
                graphs[k or node.uid] = v
    return graphs


def insert_graphs_to_atomized(full_graph: OptGraph, node_uid: str, graph: OptGraph) -> OptGraph:
    """ Insert graph to full_graph with atomized model in node with uid node_uid """
    if node_uid == '':
        full_graph = graph
    else:
        full_graph = full_graph
        # look for node with uid == node_uid
        nodes = full_graph.nodes[:]
        while nodes:
            node = nodes.pop()
            if node.uid == node_uid:
                break
            if 'pipeline' in node.content['params']:
                nodes.extend(node.content['params']['pipeline'].nodes)
        else:
            raise ValueError(f"Unknown node uid: {node_uid}")
        if 'pipeline' not in node.content['params']:
            raise ValueError(f"Cannot insert graph to non atomized model")
        node.content['params']['pipeline'] = graph
    return full_graph
