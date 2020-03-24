from copy import deepcopy
from typing import Optional, List

import networkx as nx

from core.composer.node import Node, SecondaryNode, PrimaryNode, CachedNodeResult
from core.models.data import InputData

ERROR_PREFIX = 'Invalid chain configuration:'


class Chain:
    def __init__(self):
        self.nodes = []

    def fit_from_scratch(self, input_data: InputData):
        # Clean all cache and fit all models
        self._clean_model_cache()
        print('Fit chain from scratch')
        self.root_node.fit(input_data=input_data)

    def fit(self, input_data: InputData, use_cache=False):
        if not use_cache:
            return self.fit_from_scratch(input_data=input_data)

    def predict(self, input_data: InputData):
        if not self.is_all_cache_actual():
            raise Exception('Cache is not actual')
        result = self.root_node.predict(input_data=input_data)
        return result

    def add_node(self, new_node: Node):
        """
        Append new node to chain list

        """
        self.nodes.append(new_node)

    def replace_node(self, old_node: Node, new_node: Node):
        new_node = deepcopy(new_node)
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
        g1, _ = as_nx_graph(self, True)
        g2, _ = as_nx_graph(other, True)
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

    def _flat_nodes_tree(self, node):
        raise NotImplementedError()


def as_nx_graph(chain: Chain, force_node_model_name=False):
    """force_node_model_name should be True in case when parameter mode_math of function nx.is_isomorphic is the
    function which compares nodes model names"""
    graph = nx.DiGraph()

    node_labels = {}
    for node in chain.nodes:
        id, label = node.node_id, f'{node}'
        node_labels[node.node_id] = label
        if not force_node_model_name:
            graph.add_node(id)
        else:
            graph.add_node(label)

    def add_edges(graph, chain):
        for node in chain.nodes:
            if node.nodes_from is not None:
                for child in node.nodes_from:
                    graph.add_edge(child.node_id, node.node_id)

    add_edges(graph, chain)
    return graph, node_labels


def _is_cache_actual(node, cache: CachedNodeResult):
    if cache is not None and cache.is_actual(node):
        return True

    return False


def name_comparison_func(first_node_model_name, second_node_model_name) -> bool:
    return first_node_model_name == second_node_model_name
