from typing import Optional

from core.composer.node import Node, SecondaryNode, PrimaryNode
from core.models.data import InputData, OutputData


class Chain:
    def __init__(self, base_node: Optional[Node] = None):
        if base_node is None:
            self.nodes = []
        else:
            self.nodes = self._flat_nodes_tree(base_node)

    def evaluate(self, new_data: Optional[InputData] = None) -> OutputData:
        is_train_models = True
        is_use_cache = True
        # if the chain should be evaluated for the new dataset
        if new_data is not None:
            # train models is necessary
            if any([node.cached_result is None for node in self.nodes]):
                self.evaluate()
            # update data in primary nodes
            for node in self.nodes:
                if isinstance(node, PrimaryNode):
                    node.input_data = new_data
            is_train_models = False
            is_use_cache = False
        # update flags in nodes
        for node in self.nodes:
            node.eval_strategy.is_train_models = is_train_models
            node.is_use_cache = is_use_cache
            if new_data is None:
                # preserve reference data in nodes if necessary
                node.input_data = self.reference_data
        return self.root_node.apply()

    def add_node(self, new_node: Node):
        # Append new node to chain
        self.nodes.append(new_node)

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
                if not self._is_node_has_child(node)][0]
        return root

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

        return primary_nodes[0].input_data

    @property
    def reference_data(self) -> Optional[InputData]:
        if len(self.nodes) == 0:
            return None
        primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
        assert len(primary_nodes) > 0

        return primary_nodes[0].input_data

    @reference_data.setter
    def reference_data(self, data):
        if len(self.nodes) > 0:
            primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
            for node in primary_nodes:
                node.input_data = data

    @reference_data.setter
    def reference_data(self, data):
        if len(self.nodes) > 0:
            primary_nodes = [node for node in self.nodes if isinstance(node, PrimaryNode)]
            for node in primary_nodes:
                node.input_data = data
