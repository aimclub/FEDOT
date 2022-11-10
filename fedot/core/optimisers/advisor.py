from typing import List, Any

from fedot.core.optimisers.graph import OptNode
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class RemoveType(Enum):
    """Defines allowed kinds of removals in Graph. Used by mutations."""
    node_only = 'node_only'
    with_direct_children = 'with_direct_children'
    with_parents = 'with_parents'
    forbidden = 'forbidden'


class DefaultChangeAdvisor:
    """
    Class for advising of pipeline changes during evolution
    """

    def __init__(self, task=None):
        self.task = task

    def propose_change(self, node: OptNode, possible_operations: List[Any]) -> List[Any]:
        return possible_operations

    def can_be_removed(self, node: OptNode) -> RemoveType:
        return RemoveType.node_only

    def propose_parent(self, node: OptNode, possible_operations: List[Any]) -> List[Any]:
        return possible_operations
