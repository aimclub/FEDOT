from typing import List


class NodeOperator:
    def __init__(self, node):
        self._node = node

    def distance_to_primary_level(self):
        if not self._node.nodes_from:
            return 0
        else:
            return 1 + max([next_node.distance_to_primary_level for next_node in self._node.nodes_from])

    def ordered_subnodes_hierarchy(self, visited=None) -> List['Node']:
        if visited is None:
            visited = []
        nodes = [self._node]
        if self._node.nodes_from:
            for parent in self._node.nodes_from:
                if parent not in visited:
                    nodes.extend(parent.ordered_subnodes_hierarchy(visited))

        return nodes
