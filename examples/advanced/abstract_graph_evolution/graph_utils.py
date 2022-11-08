from collections.abc import Mapping

import networkx as nx
from typing import Sequence

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.optimisers.graph import OptGraph, OptNode

_NODE_KEY = 'data'


def _graphnode_to_dict(node: GraphNode) -> dict:
    return {_NODE_KEY: node}


def _optnode_from_dict(data: Mapping) -> OptNode:
    return data[_NODE_KEY]


def nx_to_optgraph(graph: nx.DiGraph, node_adapt=_optnode_from_dict) -> OptGraph:
    # node parents (node.nodes_from) must be preserved inside nodes
    nodes = []
    mapped_nodes = {}

    def map_predecessors(node_id) -> Sequence[OptNode]:
        predecessor_nodes = []
        for pred_id in graph.predecessors(node_id):
            pred_node = mapped_nodes.get(pred_id)
            if pred_node is None:
                node_data = graph.nodes[pred_id]
                pred_node = node_adapt(node_data)
                mapped_nodes[pred_id] = pred_node
            predecessor_nodes.append(pred_node)
        return predecessor_nodes

    for node_id, node_data in graph.nodes.items():
        # transform node
        node = node_adapt(node_data)
        mapped_nodes[node_id] = node
        # append its parent edges
        node.nodes_from = map_predecessors(node_id)
        # append node
        nodes.append(node)

    return OptGraph(nodes)


def optgraph_to_nx(graph: Graph, node_restore=_graphnode_to_dict) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    node_labels = {}

    # add nodes
    for node in graph.nodes:
        node_labels[node.uid] = node_restore(node)
        nx_graph.add_node(node.uid)

    # add edges
    for node in graph.nodes:
        for parent in node.nodes_from:
            nx_graph.add_edge(parent.uid, node.uid)

    # add nodes ad labels
    nx.set_node_attributes(nx_graph, node_labels)

    return nx_graph
