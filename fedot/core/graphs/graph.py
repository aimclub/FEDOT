from copy import deepcopy
from typing import List, Optional, Union
from uuid import uuid4

from fedot.core.composer.visualisation import GraphVisualiser
from fedot.core.graphs.graph_node import GraphNode, PrimaryGraphNode
from fedot.core.graphs.graph_operator import GraphOperator
from fedot.core.log import Log, default_log

ERROR_PREFIX = 'Invalid chain configuration:'


class GraphObject:
    """
    Base class used for composite model structure definition

    :param nodes: GraphNode object(s)
    :param log: Log object to record messages
    """

    def __init__(self, nodes: Optional[Union[GraphNode, List[GraphNode]]] = None,
                 log: Log = None):
        self.uid = str(uuid4())
        self.nodes = []
        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self.operator = GraphOperator(self)

        if nodes:
            if isinstance(nodes, list):
                for node in nodes:
                    self.add_node(node)
            else:
                self.add_node(nodes)

    def add_node(self, new_node: GraphNode):
        """
        Add new node to the Chain

        :param new_node: new Node object
        """
        self.operator.add_node(new_node)

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        """
        Replace old_node with new one.

        :param old_node: GraphNode object to replace
        :param new_node: GraphNode object to replace
        """

        self.operator.update_node(old_node, new_node)

    def update_subtree(self, old_subroot: GraphNode, new_subroot: GraphNode):
        """
        Replace the subtrees with old and new nodes as subroots

        :param old_subroot: GraphNode object to replace
        :param new_subroot: GraphNode object to replace
        """
        self.operator.update_subtree(old_subroot, new_subroot)

    def delete_node(self, node: GraphNode):
        """
        Delete chosen node redirecting all its parents to the child.

        :param node: GraphNode object to delete
        """

        self.operator.delete_node(node)

    def delete_subtree(self, subroot: GraphNode):
        """
        Delete the subtree with node as subroot.

        :param subroot:
        """
        self.operator.delete_subtree(subroot)

    def show(self, path: str = None):
        GraphVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        if isinstance(self.root_node, list):
            if isinstance(other.root_node, list):
                return set([rn.descriptive_id for rn in self.root_node]) == \
                       set([rn.descriptive_id for rn in other.root_node])
            else:
                return False
        elif isinstance(other.root_node, list):
            return False
        else:
            return self.root_node.descriptive_id == other.root_node.descriptive_id

    def __str__(self):
        description = {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }
        return f'{description}'

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self):
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not any(self.operator.node_children(node))]
        if len(root) == 1:
            return root[0]
        return root

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryGraphNode):
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        if isinstance(self.root_node, list):
            return max([_depth_recursive(n) for n in self.root_node])
        else:
            return _depth_recursive(self.root_node)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.uid = uuid4()
        return result

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result.uid = uuid4()
        return result
