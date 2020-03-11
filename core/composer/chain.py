from copy import deepcopy
from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import InputData, OutputData
from core.repository.node_types import SecondaryNodeType
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
        if isinstance(new_node, SecondaryNode) and new_node.status == SecondaryNodeType.terminal:
            self._self_validation()

    def update_node(self, new_node: Node):
        raise NotImplementedError()

    def _is_node_has_child(self, node):
        return any([(node in other_node.nodes_from)
                    for other_node in self.nodes if isinstance(other_node, SecondaryNode)])

    def _self_validation(self):
        has_one_root(self)
        has_no_cycle(self)
        has_no_self_cycled_nodes(self)
        has_no_isolated_nodes(self)
        has_primary_nodes(self)

    @property
    def root_node(self) -> Optional[Node]:
        global ERROR_PREFIX
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


def has_one_root(chain: Chain):
    if chain.root_node:
        return True


def has_no_cycle(chain: Chain):
    visited = {node.node_id: False for node in chain.nodes}
    root = chain.root_node
    if visited[root.node_id] is False:
        traverse(root, visited)


def has_no_isolated_nodes(chain: Chain):
    visited = {node.node_id: False for node in chain.nodes}
    root = chain.root_node
    num_visited_nodes = 0
    global ERROR_PREFIX
    if visited[root.node_id] is False:
        num_visited_nodes = traverse(root, visited)
    if num_visited_nodes < len(chain.nodes):
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated nodes')


def has_primary_nodes(chain: Chain):
    return any(node for node in chain.nodes if isinstance(node, PrimaryNode))


def has_no_self_cycled_nodes(chain: Chain):
    global ERROR_PREFIX
    if any([node for node in chain.nodes if isinstance(node, SecondaryNode) and node in node.nodes_from]):
        raise ValueError(f'{ERROR_PREFIX} Chain has self-cycled nodes')


def traverse(node: Optional[Node], visited: dict):
    """
    dfs-algorithm

    :param node: OPtional[Node]
    :param visited: dict{node.id: bool}
    :return: int
    """

    visited[node.node_id] = True
    visited_nodes = 1
    global ERROR_PREFIX
    if isinstance(node, SecondaryNode):
        for parent in node.nodes_from:
            if not visited[parent.node_id]:
                visited_nodes += traverse(parent, visited)
            else:
                raise ValueError(f'{ERROR_PREFIX} Chain has a cycle')
        is_visited = False
    else:
        is_visited = False

    visited[node.node_id] = is_visited
    return visited_nodes
