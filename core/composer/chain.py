from core.data import Data
from core.node import Node


class Chain:
    def __init__(self):
        self.nodes = []

    def evaluate(self) -> Data:
        raise NotImplementedError()

    def add_node(self, new_node: Node):
        # Append new node to chain
        raise NotImplementedError()

    def update_node(self, new_node: Node):
        raise NotImplementedError()
