from typing import Optional, List

import networkx as nx

from core.composer.node import Node, SecondaryNode, PrimaryNode, CachedNodeResult
from core.models.data import InputData
from copy import deepcopy

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

    def replace_node(self, old_node: Node, new_node: Node):
        new_node = new_node.duplicate
        old_node_offspring = self._node_childs(old_node)
        for old_node_child in old_node_offspring:
            old_node_child.nodes_from[old_node_child.nodes_from.index(old_node)] = new_node
        new_nodes = [parent for parent in new_node.subtree_nodes if not parent in self.nodes]
        old_nodes = [node for node in self.nodes if not node in old_node.subtree_nodes]
        self.nodes = new_nodes + old_nodes

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    def _clean_model_cache(self):
        for node in self.nodes:
            node.cached_result = None

    def is_all_cache_actual(self):
        cache_status = [_is_cache_actual(node, node.cached_result) for node in self.nodes]
        return all(cache_status)

    def _node_childs(self, node) -> List[Optional[Node]]:
        return [other_node for other_node in self.nodes if isinstance(other_node, SecondaryNode) if
                node in other_node.nodes_from]

    def _is_node_has_child(self, node) -> bool:
        return any(self._node_childs(node))

    def __eq__(self, other) -> bool:
        g1, _ = as_nx_graph(self)
        g2, _ = as_nx_graph(other)
        return nx.is_isomorphic(g1, g2, node_match=name_comparison_func)

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

    @property
    def duplicate(self) -> 'Chain':
        duplicate_chain = deepcopy(self)
        root_node_duplicate = self.root_node.duplicate
        duplicate_chain.nodes = root_node_duplicate.subtree_nodes
        return duplicate_chain

    def _flat_nodes_tree(self, node):
        raise NotImplementedError()


def as_nx_graph(chain: Chain) -> (nx.DiGraph, dict):
    """force_node_model_name should be True in case when parameter mode_math of function nx.is_isomorphic is the
    function which compares nodes model names"""
    graph = nx.DiGraph()

    node_labels = {}
    for node in chain.nodes:
        id, label = node.node_id, f'{node}'
        node_labels[node.node_id] = label
        graph.add_node(id)
    nx.set_node_attributes(graph, node_labels, 'model')

    def add_edges(graph, chain):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(child.node_id, node.node_id)

    add_edges(graph, chain)
    return graph, node_labels


def _is_cache_actual(node, cache: CachedNodeResult) -> bool:
    if cache is not None and cache.is_actual(node):
        return True
    return False


def name_comparison_func(first_node_model_name, second_node_model_name) -> bool:
    return first_node_model_name['model'] == second_node_model_name['model']
