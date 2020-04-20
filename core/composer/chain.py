from copy import deepcopy
from typing import Optional, List
from uuid import uuid4

import networkx as nx

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import InputData

ERROR_PREFIX = 'Invalid chain configuration:'


class Chain:
    def __init__(self):
        self.nodes = []

    def fit_from_scratch(self, input_data: InputData, verbose=False):
        # Clean all cache and fit all models
        print('Fit chain from scratch')
        self.fit(input_data, use_cache=False, verbose=verbose)

    def fit(self, input_data: InputData, use_cache=True, verbose=False):
        if not use_cache:
            self._clean_model_cache()
        train_predicted = self.root_node.fit(input_data=input_data, verbose=verbose)

        return train_predicted

    def predict(self, input_data: InputData):
        if not self.is_all_cache_actual():
            raise Exception('Trained model cache is not actual or empty')
        result = self.root_node.predict(input_data=input_data)
        return result

    def add_node(self, new_node: Node):
        """
        Append new node to chain list

        """
        self.nodes.append(new_node)

    def _actualise_old_node_childs(self, old_node: Node, new_node: Node):
        old_node_offspring = self._node_childs(old_node)
        for old_node_child in old_node_offspring:
            old_node_child.nodes_from[old_node_child.nodes_from.index(old_node)] = new_node

    def replace_node_with_parents(self, old_node: Node, new_node: Node):
        new_node = deepcopy(new_node)
        self._actualise_old_node_childs(old_node, new_node)
        new_nodes = [parent for parent in new_node.subtree_nodes if not parent in self.nodes]
        old_nodes = [node for node in self.nodes if not node in old_node.subtree_nodes]
        self.nodes = new_nodes + old_nodes

    def update_node(self, old_node: Node, new_node: Node):
        self._actualise_old_node_childs(old_node, new_node)
        new_node.nodes_from = old_node.nodes_from
        self.nodes.remove(old_node)
        self.nodes.append(new_node)

    def _clean_model_cache(self):
        for node in self.nodes:
            node.cache.clear()

    def is_all_cache_actual(self):
        cache_status = [node.cache.actual_cached_model is not None for node in self.nodes]
        return all(cache_status)

    def _node_childs(self, node) -> List[Optional[Node]]:
        return [other_node for other_node in self.nodes if isinstance(other_node, SecondaryNode) if
                node in other_node.nodes_from]

    def _is_node_has_child(self, node) -> bool:
        return any(self._node_childs(node))

    def __eq__(self, other) -> bool:
        return self.root_node.descriptive_id == other.root_node.descriptive_id

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
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
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


def as_nx_graph(chain: Chain):
    graph = nx.DiGraph()

    node_labels = {}
    new_node_idx = {}
    for node in chain.nodes:
        unique_id, label = uuid4(), str(node)
        node_labels[unique_id] = str(node)
        new_node_idx[node] = unique_id
        graph.add_node(unique_id)

    def add_edges(graph, chain, new_node_idx):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(new_node_idx[child], new_node_idx[node])

    add_edges(graph, chain, new_node_idx)
    return graph, node_labels
