from copy import deepcopy
from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import InputData, OutputData


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
            node.input_data = deepcopy(self.reference_data)
        return self.root_node.apply()

    def predict(self, new_data: InputData) -> OutputData:
        if any([node.cached_result is None for node in self.nodes]):
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

    def validate_chain(self):
        self.root_node
        self._has_no_cycle()
        self._has_no_self_cycled_nodes()
        self._has_isolated_nodes()
        self._has_primary_nodes()

    def _has_no_cycle(self):
        visited = {node.node_id: False for node in self.nodes}
        root = self.root_node
        if visited[root.node_id] is False:
            _explore(root, visited)

    def _has_isolated_nodes(self):
        visited = {node.node_id: False for node in self.nodes}
        root = self.root_node
        num_visited_nodes = 0
        if visited[root.node_id] is False:
            num_visited_nodes = _explore(root, visited, check_isolated=True)
        if num_visited_nodes < len(self.nodes):
            raise ValueError('Invalid chain configuration: Chain has isolated nodes')

    def _has_primary_nodes(self):
        return any(node for node in self.nodes if isinstance(node, PrimaryNode))

    def _has_no_self_cycled_nodes(self):
        if any([node for node in self.nodes if isinstance(node, SecondaryNode) and node in node.nodes_from]):
            raise ValueError('Invalid chain configuration: Chain has self-cycled nodes')

    @property
    def root_node(self) -> Optional[Node]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not self._is_node_has_child(node)]
        assert len(root) <= 1, 'More than 1 root_nodes in chain'
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


def _explore(node, visit, check_isolated=False):
    """
    dfs-algorithm

    :param node: OPtional[Node]
    :param visit: dict{node.id: bool}
    :param check_isolated: bool
    :return: int if check_isolated is True else void
    """
    visit[node.node_id] = True
    visited_nodes = 1
    if isinstance(node, SecondaryNode):
        for parent in node.nodes_from:
            if not visit[parent.node_id]:
                if check_isolated:
                    visited_nodes += _explore(parent, visit, check_isolated=True)
                else:
                    _explore(parent, visit)
            else:
                raise ValueError('Invalid chain configuration: Chain has a cycle')
        visit[node.node_id] = False
    else:
        visit[node.node_id] = False

    if check_isolated:
        return visited_nodes
