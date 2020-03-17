from copy import deepcopy
from typing import Optional

import networkx as nx

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import InputData, OutputData

ERROR_PREFIX = 'Invalid chain configuration:'


class Chain:
    def __init__(self):
        self.nodes = []
        self.reference_data = None

    def train(self) -> OutputData:
        # if the chain should be evaluated for the new dataset
        for node in self.nodes:
            node.eval_strategy.is_train_models = True
            node.is_caching = True
            # set reference data in nodes
            if isinstance(node, PrimaryNode):
                node.input_data = deepcopy(self.reference_data)
        return self.root_node.apply()

    def predict(self, new_data: InputData) -> OutputData:
        if any([(node.cached_result is None) or (not node.cached_result.is_actual(node.nodes_from))
                for node in self.nodes]):
            self.train()
            # update data in primary nodes
        for node in self.nodes:
            if isinstance(node, PrimaryNode):
                node.input_data = deepcopy(new_data)
        # update flags in nodes
        for node in self.nodes:
            node.eval_strategy.is_train_models = False
            node.is_caching = False
        return self.root_node.apply()

    def add_node(self, new_node: Node):
        """
        Append new node to chain list

        """
        self.nodes.append(new_node)
        if isinstance(new_node, PrimaryNode):
            # TODO refactor
            self.reference_data = deepcopy(new_node.input_data)

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    def _is_node_has_child(self, node):
        return any([(node in other_node.nodes_from)
                    for other_node in self.nodes if isinstance(other_node, SecondaryNode)])

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not self._is_node_has_child(node)]
        if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in chain')
        return root[0]

    @property
    def length(self):
        return len(self.nodes)

    @property
    def depth(self):
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)

    def _flat_nodes_tree(self, node):
        raise NotImplementedError()

    @property
    def reference_data(self) -> Optional[InputData]:
        if len(self.nodes) == 0:
            return None
        primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
        assert len(primary_nodes) > 0

        return deepcopy(primary_nodes[0].input_data)

    @reference_data.setter
    def reference_data(self, data):
        if len(self.nodes) > 0:
            primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
            for node in primary_nodes:
                node.input_data = deepcopy(data)


def as_nx_graph(chain: Chain):
    graph = nx.DiGraph()

    node_labels = {}
    for node in chain.nodes:
        graph.add_node(node.node_id)
        node_labels[node.node_id] = f'{node}'

    def add_edges(graph, chain):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(child.node_id, node.node_id)

    add_edges(graph, chain)
    return graph, node_labels
