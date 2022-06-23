from copy import copy
from typing import Union, Sequence, List, Callable

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.utilities.data_structures import Copyable


class GraphDelegate(Graph, Copyable):
    """
    Graph that delegates calls to another Graph implementation.

    The class implements idiom 'inheritance through composition',
    and its purpose is the cleaner code organisation:
    - Avoid inheriting from specific Graph implementations
    - Strictly hide Graph implementation details from inheritors.

    :param delegate: Graph implementation to delegate to.
    """

    def __init__(self, delegate: Graph):
        self.operator = delegate

    @classmethod
    def default(cls, *args, **kwargs) -> Graph:
        """Returns GraphDelegate instance with default delegate.
        Function arguments are passed to its constructor."""
        impl_factory: Callable[..., Graph] = GraphOperator
        return cls(impl_factory(*args, **kwargs))

    def add_node(self, new_node: 'GraphNode'):
        self.operator.add_node(new_node)

    def update_node(self, old_node: 'GraphNode', new_node: 'GraphNode'):
        self.operator.update_node(old_node, new_node)

    def update_subtree(self, old_subroot: 'GraphNode', new_subroot: 'GraphNode'):
        self.operator.update_subtree(old_subroot, new_subroot)

    def delete_node(self, node: 'GraphNode'):
        self.operator.delete_node(node)

    def delete_subtree(self, subroot: 'GraphNode'):
        self.operator.delete_subtree(subroot)

    def __eq__(self, other) -> bool:
        return self.operator.__eq__(other)

    @property
    def root_node(self) -> Union['GraphNode', Sequence['GraphNode']]:
        return self.operator.root_node

    @property
    def nodes(self) -> List['GraphNode']:
        return self.operator.nodes

    @nodes.setter
    def nodes(self, new_nodes: List['GraphNode']):
        self.operator.nodes = new_nodes

    @property
    def depth(self) -> int:
        return self.operator.depth

    @property
    def length(self) -> int:
        return self.operator.length

    @property
    def descriptive_id(self) -> str:
        return self.operator.descriptive_id

    def __str__(self):
        return self.operator.__str__()

    def __repr__(self):
        return self.operator.__repr__()

    def __copy__(self):
        """Delegates copy to underlying graph operator."""
        result = Copyable.__copy__(self)
        result.operator = copy(self.operator)
        return result
